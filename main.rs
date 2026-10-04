//! PRISM - a zero-dependency, multithreaded path tracer.
//!
//! Features: diffuse / metal / glass / emissive materials, depth of field,
//! gamma-correct output, a hand-written PNG encoder, and a procedural scene.
//!
//! Usage: prism [width] [samples_per_pixel] [output.png]

use std::fs::File;
use std::io::Write;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// ───────────────────────────── Math ─────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

const fn v(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: Vec3) -> Vec3 {
        v(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn len2(self) -> f64 {
        self.dot(self)
    }
    fn len(self) -> f64 {
        self.len2().sqrt()
    }
    fn unit(self) -> Vec3 {
        self / self.len()
    }
    fn near_zero(self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }
    fn reflect(self, n: Vec3) -> Vec3 {
        self - n * (2.0 * self.dot(n))
    }
    fn refract(self, n: Vec3, eta_ratio: f64) -> Vec3 {
        let cos_t = (-self).dot(n).min(1.0);
        let perp = (self + n * cos_t) * eta_ratio;
        let par = n * -((1.0 - perp.len2()).abs().sqrt());
        perp + par
    }
    fn hadamard(self, o: Vec3) -> Vec3 {
        v(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}

impl Add for Vec3 { type Output = Vec3; fn add(self, o: Vec3) -> Vec3 { v(self.x + o.x, self.y + o.y, self.z + o.z) } }
impl Sub for Vec3 { type Output = Vec3; fn sub(self, o: Vec3) -> Vec3 { v(self.x - o.x, self.y - o.y, self.z - o.z) } }
impl Mul<f64> for Vec3 { type Output = Vec3; fn mul(self, t: f64) -> Vec3 { v(self.x * t, self.y * t, self.z * t) } }
impl Div<f64> for Vec3 { type Output = Vec3; fn div(self, t: f64) -> Vec3 { self * (1.0 / t) } }
impl Neg for Vec3 { type Output = Vec3; fn neg(self) -> Vec3 { v(-self.x, -self.y, -self.z) } }

// ───────────────────────────── RNG ──────────────────────────────

/// Tiny xorshift64* generator: fast, deterministic, no dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    fn in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let p = v(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            if p.len2() < 1.0 {
                return p;
            }
        }
    }
    fn unit_vector(&mut self) -> Vec3 {
        self.in_unit_sphere().unit()
    }
    fn in_unit_disk(&mut self) -> Vec3 {
        loop {
            let p = v(self.range(-1.0, 1.0), self.range(-1.0, 1.0), 0.0);
            if p.len2() < 1.0 {
                return p;
            }
        }
    }
}

// ─────────────────────────── Ray & Scene ────────────────────────

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Ray {
    fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone, Copy)]
enum Material {
    Diffuse(Vec3),
    Metal { albedo: Vec3, fuzz: f64 },
    Glass(f64),
    Light(Vec3),
}

struct Sphere {
    center: Vec3,
    radius: f64,
    mat: Material,
}

struct Hit {
    t: f64,
    point: Vec3,
    normal: Vec3,
    front: bool,
    mat: Material,
}

impl Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.dir.len2();
        let half_b = oc.dot(r.dir);
        let c = oc.len2() - self.radius * self.radius;
        let disc = half_b * half_b - a * c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let mut t = (-half_b - sq) / a;
        if t < t_min || t > t_max {
            t = (-half_b + sq) / a;
            if t < t_min || t > t_max {
                return None;
            }
        }
        let point = r.at(t);
        let outward = (point - self.center) / self.radius;
        let front = r.dir.dot(outward) < 0.0;
        let normal = if front { outward } else { -outward };
        Some(Hit { t, point, normal, front, mat: self.mat })
    }
}

fn schlick(cos: f64, ratio: f64) -> f64 {
    let r0 = ((1.0 - ratio) / (1.0 + ratio)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos).powi(5)
}

/// Returns (attenuation, scattered ray) or None if the ray is absorbed.
fn scatter(r: &Ray, h: &Hit, rng: &mut Rng) -> Option<(Vec3, Ray)> {
    match h.mat {
        Material::Diffuse(albedo) => {
            let mut d = h.normal + rng.unit_vector();
            if d.near_zero() {
                d = h.normal;
            }
            Some((albedo, Ray { origin: h.point, dir: d }))
        }
        Material::Metal { albedo, fuzz } => {
            let refl = r.dir.unit().reflect(h.normal) + rng.in_unit_sphere() * fuzz;
            if refl.dot(h.normal) > 0.0 {
                Some((albedo, Ray { origin: h.point, dir: refl }))
            } else {
                None
            }
        }
        Material::Glass(ior) => {
            let ratio = if h.front { 1.0 / ior } else { ior };
            let u = r.dir.unit();
            let cos = (-u).dot(h.normal).min(1.0);
            let sin = (1.0 - cos * cos).sqrt();
            let dir = if ratio * sin > 1.0 || schlick(cos, ratio) > rng.f() {
                u.reflect(h.normal)
            } else {
                u.refract(h.normal, ratio)
            };
            Some((v(1.0, 1.0, 1.0), Ray { origin: h.point, dir }))
        }
        Material::Light(_) => None,
    }
}

fn closest_hit(world: &[Sphere], r: &Ray) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    let mut closest = f64::INFINITY;
    for s in world {
        if let Some(h) = s.hit(r, 1e-3, closest) {
            closest = h.t;
            best = Some(h);
        }
    }
    best
}

fn sky(r: &Ray) -> Vec3 {
    let t = 0.5 * (r.dir.unit().y + 1.0);
    // Dim twilight gradient so the emissive lights stand out.
    (v(0.04, 0.05, 0.10) * (1.0 - t) + v(0.18, 0.26, 0.45) * t) * 0.8
}

/// Iterative path tracing (no recursion, no stack growth).
fn trace(world: &[Sphere], mut ray: Ray, max_depth: u32, rng: &mut Rng) -> Vec3 {
    let mut throughput = v(1.0, 1.0, 1.0);
    let mut radiance = v(0.0, 0.0, 0.0);
    for _ in 0..max_depth {
        match closest_hit(world, &ray) {
            None => {
                radiance = radiance + throughput.hadamard(sky(&ray));
                break;
            }
            Some(h) => {
                if let Material::Light(e) = h.mat {
                    radiance = radiance + throughput.hadamard(e);
                    break;
                }
                match scatter(&ray, &h, rng) {
                    Some((att, next)) => {
                        throughput = throughput.hadamard(att);
                        ray = next;
                        // Russian roulette keeps long paths cheap.
                        let p = throughput.x.max(throughput.y).max(throughput.z).clamp(0.05, 0.95);
                        if rng.f() > p {
                            break;
                        }
                        throughput = throughput / p;
                    }
                    None => break,
                }
            }
        }
    }
    radiance
}

// ───────────────────────────── Camera ───────────────────────────

struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(from: Vec3, at: Vec3, up: Vec3, vfov: f64, aspect: f64, aperture: f64, focus: f64) -> Self {
        let h = (vfov.to_radians() / 2.0).tan();
        let vh = 2.0 * h;
        let vw = aspect * vh;
        let w = (from - at).unit();
        let u = up.cross(w).unit();
        let vv = w.cross(u);
        let horizontal = u * (vw * focus);
        let vertical = vv * (vh * focus);
        Camera {
            origin: from,
            horizontal,
            vertical,
            lower_left: from - horizontal / 2.0 - vertical / 2.0 - w * focus,
            u,
            v: vv,
            lens_radius: aperture / 2.0,
        }
    }

    fn ray(&self, s: f64, t: f64, rng: &mut Rng) -> Ray {
        let rd = rng.in_unit_disk() * self.lens_radius;
        let off = self.u * rd.x + self.v * rd.y;
        Ray {
            origin: self.origin + off,
            dir: self.lower_left + self.horizontal * s + self.vertical * t - self.origin - off,
        }
    }
}

// ───────────────────────────── Scene ────────────────────────────

fn build_scene() -> Vec<Sphere> {
    let mut rng = Rng::new(2024);
    let mut w = Vec::new();

    // Ground
    w.push(Sphere { center: v(0.0, -1000.0, 0.0), radius: 1000.0, mat: Material::Diffuse(v(0.32, 0.33, 0.36)) });

    // Procedural field of small spheres
    for a in -9..9 {
        for b in -9..9 {
            let center = v(a as f64 + 0.8 * rng.f(), 0.2, b as f64 + 0.8 * rng.f());
            if (center - v(0.0, 0.2, 0.0)).len() < 2.2
                || (center - v(-4.0, 0.2, 0.0)).len() < 1.4
                || (center - v(4.0, 0.2, 0.0)).len() < 1.4
            {
                continue;
            }
            let pick = rng.f();
            let mat = if pick < 0.45 {
                let c = v(rng.f() * rng.f(), rng.f() * rng.f(), rng.f() * rng.f());
                Material::Diffuse(c)
            } else if pick < 0.70 {
                Material::Metal {
                    albedo: v(rng.range(0.5, 1.0), rng.range(0.5, 1.0), rng.range(0.5, 1.0)),
                    fuzz: rng.range(0.0, 0.35),
                }
            } else if pick < 0.88 {
                Material::Glass(1.5)
            } else {
                // Glowing orbs
                let hue = rng.f();
                let c = v(
                    0.5 + 0.5 * (6.283 * hue).cos(),
                    0.5 + 0.5 * (6.283 * (hue + 0.33)).cos(),
                    0.5 + 0.5 * (6.283 * (hue + 0.66)).cos(),
                );
                Material::Light(c * 5.0)
            };
            w.push(Sphere { center, radius: 0.2, mat });
        }
    }

    // Hero spheres
    w.push(Sphere { center: v(0.0, 1.0, 0.0), radius: 1.0, mat: Material::Glass(1.5) });
    w.push(Sphere { center: v(0.0, 1.0, 0.0), radius: -0.9, mat: Material::Glass(1.5) }); // hollow shell
    w.push(Sphere { center: v(-4.0, 1.0, 0.0), radius: 1.0, mat: Material::Diffuse(v(0.55, 0.15, 0.12)) });
    w.push(Sphere { center: v(4.0, 1.0, 0.0), radius: 1.0, mat: Material::Metal { albedo: v(0.85, 0.75, 0.55), fuzz: 0.0 } });
    // Big warm key light floating above
    w.push(Sphere { center: v(-3.0, 7.0, 4.0), radius: 2.0, mat: Material::Light(v(14.0, 11.0, 8.0)) });
    w
}

// ───────────────────────────── PNG ──────────────────────────────

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut body = kind.to_vec();
    body.extend_from_slice(data);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// Writes an 8-bit RGB PNG using uncompressed ("stored") deflate blocks.
fn write_png(path: &str, w: usize, h: usize, rgb: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((w * 3 + 1) * h);
    for y in 0..h {
        raw.push(0); // filter: none
        raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    File::create(path)?.write_all(&out)
}

// ───────────────────────────── Main ─────────────────────────────

fn aces(x: f64) -> f64 {
    // Narkowicz ACES filmic tone-mapping approximation.
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(960);
    let spp: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let out_path = args.get(3).cloned().unwrap_or_else(|| "render.png".into());
    let height = width * 9 / 16;
    let max_depth = 40;

    let world = Arc::new(build_scene());
    let from = v(11.0, 2.6, 6.5);
    let at = v(0.0, 0.7, 0.0);
    let cam = Arc::new(Camera::new(from, at, v(0.0, 1.0, 0.0), 28.0, width as f64 / height as f64, 0.12, (from - at).len()));

    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("PRISM | {}x{} | {} spp | {} threads | {} spheres", width, height, spp, threads, world.len());

    let image = Arc::new(Mutex::new(vec![0u8; width * height * 3]));
    let next_row = Arc::new(AtomicUsize::new(0));
    let done_rows = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let (world, cam, image, next_row, done_rows) =
                (world.clone(), cam.clone(), image.clone(), next_row.clone(), done_rows.clone());
            thread::spawn(move || loop {
                let y = next_row.fetch_add(1, Ordering::Relaxed);
                if y >= height {
                    break;
                }
                let mut rng = Rng::new((y as u64) * 7919 + tid as u64 + 1);
                let mut row = vec![0u8; width * 3];
                for x in 0..width {
                    let mut c = v(0.0, 0.0, 0.0);
                    for _ in 0..spp {
                        let s = (x as f64 + rng.f()) / (width - 1) as f64;
                        let t = ((height - 1 - y) as f64 + rng.f()) / (height - 1) as f64;
                        let mut sample = trace(&world, cam.ray(s, t, &mut rng), max_depth, &mut rng);
                        // Firefly suppression: clamp extreme single-sample energy.
                        let m = sample.x.max(sample.y).max(sample.z);
                        if m > 6.0 {
                            sample = sample * (6.0 / m);
                        }
                        c = c + sample;
                    }
                    c = c / spp as f64;
                    let px = [aces(c.x), aces(c.y), aces(c.z)];
                    for (i, p) in px.iter().enumerate() {
                        row[x * 3 + i] = (p.powf(1.0 / 2.2) * 255.0 + 0.5) as u8; // gamma 2.2
                    }
                }
                image.lock().unwrap()[y * width * 3..(y + 1) * width * 3].copy_from_slice(&row);
                let d = done_rows.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 20 == 0 || d == height {
                    eprint!("\r  progress: {:>3}%", d * 100 / height);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    eprintln!();

    let img = image.lock().unwrap();
    write_png(&out_path, width, height, &img).expect("failed to write PNG");
    println!("Done in {:.1}s -> {}", start.elapsed().as_secs_f64(), out_path);
}
