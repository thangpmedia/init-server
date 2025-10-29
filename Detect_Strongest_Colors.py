# Palette & Two Strongest Colors — css3x2 bank + CIEDE2000 + 10-color 60/30/10 palette
# Save as: ComfyUI/custom_nodes/Detect_Strongest_Colors.py

import torch
import numpy as np
import json
import math

# -------------------------
# CSS3 named colors (base)
# -------------------------
CSS3_COLORS = {
    "black": (0,0,0), "dimgray": (105,105,105), "gray": (128,128,128),
    "darkgray": (169,169,169), "silver": (192,192,192), "gainsboro": (220,220,220),
    "whitesmoke": (245,245,245), "white": (255,255,255),
    "maroon": (128,0,0), "darkred": (139,0,0), "firebrick": (178,34,34),
    "crimson": (220,20,60), "red": (255,0,0), "tomato": (255,99,71), "orangered": (255,69,0),
    "saddlebrown": (139,69,19), "sienna": (160,82,45), "chocolate": (210,105,30),
    "peru": (205,133,63), "sandybrown": (244,164,96), "burlywood": (222,184,135), "tan": (210,180,140),
    "darkorange": (255,140,0), "orange": (255,165,0), "gold": (255,215,0),
    "olive": (128,128,0), "olivedrab": (107,142,35), "darkolivegreen": (85,107,47),
    "yellowgreen": (154,205,50), "chartreuse": (127,255,0), "greenyellow": (173,255,47),
    "darkgreen": (0,100,0), "green": (0,128,0), "seagreen": (46,139,87),
    "mediumseagreen": (60,179,113), "limegreen": (50,205,50), "lime": (0,255,0),
    "forestgreen": (34,139,34), "springgreen": (0,255,127),
    "teal": (0,128,128), "darkcyan": (0,139,139), "lightseagreen": (32,178,170),
    "turquoise": (64,224,208), "mediumturquoise": (72,209,204), "aquamarine": (127,255,212),
    "cyan": (0,255,255), "lightcyan": (224,255,255),
    "navy": (0,0,128), "darkblue": (0,0,139), "mediumblue": (0,0,205),
    "blue": (0,0,255), "royalblue": (65,105,225), "dodgerblue": (30,144,255),
    "deepskyblue": (0,191,255), "lightskyblue": (135,206,250), "skyblue": (135,206,235),
    "steelblue": (70,130,180), "lightsteelblue": (176,196,222),
    "indigo": (75,0,130), "purple": (128,0,128), "darkmagenta": (139,0,139),
    "magenta": (255,0,255), "orchid": (218,112,214), "mediumorchid": (186,85,211),
    "plum": (221,160,221), "violet": (238,130,238),
    "deeppink": (255,20,147), "hotpink": (255,105,180), "palevioletred": (219,112,147),
    "pink": (255,192,203), "lightpink": (255,182,193),
    "brown": (165,42,42), "rosybrown": (188,143,143), "indianred": (205,92,92),
    "salmon": (250,128,114), "darksalmon": (233,150,122), "lightsalmon": (255,160,122),
    "khaki": (240,230,140), "darkkhaki": (189,183,107), "beige": (245,245,220),
    "cornsilk": (255,248,220), "wheat": (245,222,179), "bisque": (255,228,196),
    "moccasin": (255,228,181), "navajowhite": (255,222,173), "peachpuff": (255,218,185)
}

# -------------------------
# Color space utils
# -------------------------
def _pivot_rgb(u): 
    return np.where(u > 0.04045, ((u + 0.055)/1.055)**2.4, u/12.92)

def _rgb_to_xyz(rgb):
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    r,g,b = _pivot_rgb(r), _pivot_rgb(g), _pivot_rgb(b)
    x = r*0.4124564 + g*0.3575761 + b*0.1804375
    y = r*0.2126729 + g*0.7151522 + b*0.0721750
    z = r*0.0193339 + g*0.1191920 + b*0.9503041
    return np.stack([x,y,z], -1)

def _pivot_xyz(t):
    eps, kappa = 216/24389, 24389/27
    return np.where(t > eps, np.cbrt(t), (kappa*t + 16)/116)

def _rgb_to_lab(rgb_uint8):
    rgb = rgb_uint8.astype(np.float32)/255.0
    xyz = _rgb_to_xyz(rgb)
    Xn,Yn,Zn = 0.95047, 1.0, 1.08883
    f = _pivot_xyz(np.stack([xyz[...,0]/Xn, xyz[...,1]/Yn, xyz[...,2]/Zn], -1))
    L = 116*f[...,1] - 16
    a = 500*(f[...,0]-f[...,1])
    b = 200*(f[...,1]-f[...,2])
    return np.stack([L,a,b], -1)

def _lab_to_rgb_uint8(lab):
    # lab shape (...,3)
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    # Lab -> XYZ
    fy = (L + 16.0)/116.0
    fx = fy + (a/500.0)
    fz = fy - (b/200.0)
    Xn,Yn,Zn = 0.95047, 1.0, 1.08883

    def _f_inv(t):
        t3 = t**3
        return np.where(t3 > 216/24389, t3, (116*t - 16)/24389*27)

    x = Xn * _f_inv(fx)
    y = Yn * _f_inv(fy)
    z = Zn * _f_inv(fz)

    # XYZ -> linear RGB
    r =  3.2404542*x - 1.5371385*y - 0.4985314*z
    g = -0.9692660*x + 1.8760108*y + 0.0415560*z
    b =  0.0556434*x - 0.2040259*y + 1.0572252*z

    def _gamma_inv(u):
        return np.where(u <= 0.0031308, 12.92*u, 1.055*(np.clip(u,0,None)**(1/2.4)) - 0.055)

    r,g,b = _gamma_inv(r), _gamma_inv(g), _gamma_inv(b)
    rgb = np.stack([r,g,b], -1)
    rgb = np.clip(np.round(rgb*255.0), 0, 255).astype(np.uint8)
    return rgb

def _rgb_to_hex(rgb_uint8_triplet):
    r,g,b = [int(x) for x in rgb_uint8_triplet]
    return "#{:02X}{:02X}{:02X}".format(r,g,b)

def _rgb_to_hsv_deg(rgb_uint8_triplet):
    r,g,b = [int(x)/255.0 for x in rgb_uint8_triplet]
    mx, mn = max(r,g,b), min(r,g,b)
    d = mx - mn
    if d == 0:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / d) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / d) + 120) % 360
    else:
        h = (60 * ((r - g) / d) + 240) % 360
    s = 0.0 if mx == 0 else d / mx
    v = mx
    return h, s, v

# -------------------------
# ΔE2000 (CIEDE2000)
# -------------------------
def _delta_e_ciede2000(lab1, lab2):
    # lab1, lab2 are (...,3)
    L1, a1, b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2, a2, b2 = lab2[...,0], lab2[...,1], lab2[...,2]

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7 + 1e-12)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p*a1p + b1*b1)
    C2p = np.sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2.0 * np.sqrt(C1p*C2p) * np.sin(np.radians(dhp/2.0))

    avg_hp = (h1p + h2p) / 2.0
    avg_hp = np.where(np.abs(h1p - h2p) > 180, avg_hp + 180, avg_hp) % 360.0

    T = 1 - 0.17*np.cos(np.radians(avg_hp - 30)) + 0.24*np.cos(np.radians(2*avg_hp)) \
          + 0.32*np.cos(np.radians(3*avg_hp + 6)) - 0.20*np.cos(np.radians(4*avg_hp - 63))

    Sl = 1 + ((0.015*(avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2))
    Sc = 1 + 0.045*avg_Cp
    Sh = 1 + 0.015*avg_Cp*T

    Rt = -2*np.sqrt((avg_Cp**7)/(avg_Cp**7 + 25**7 + 1e-12)) \
         * np.sin(np.radians(60*np.exp(-((avg_hp - 275)/25)**2)))

    dE = np.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh))
    return dE

# -------------------------
# Build color banks
# -------------------------
_base_names = list(CSS3_COLORS.keys())
_base_rgbs  = np.array([CSS3_COLORS[n] for n in _base_names], dtype=np.uint8)
_base_labs  = _rgb_to_lab(_base_rgbs)

def _build_css3x2(delta_L=12.0):
    names = []
    rgbs  = []
    labs  = []
    for i, name in enumerate(_base_names):
        rgb = _base_rgbs[i]
        lab = _base_labs[i]
        names.append(name); rgbs.append(rgb); labs.append(lab)

        L,a,b = lab.tolist()
        if L >= 50.0:
            L2 = max(0.0, L - delta_L)
            suffix = "-dark"
        else:
            L2 = min(100.0, L + delta_L)
            suffix = "-light"

        lab2 = np.array([L2, a, b], dtype=np.float32)
        rgb2 = _lab_to_rgb_uint8(lab2[None,:])[0]
        names.append(name + suffix); rgbs.append(rgb2); labs.append(_rgb_to_lab(rgb2[None,:])[0])
    return names, np.array(rgbs, dtype=np.uint8), np.array(labs, dtype=np.float32)

def _nearest_named_color(rgb_uint8_triplet, bank_mode="css3x2", use_ciede2000=True, delta_L=12.0):
    # choose bank
    if bank_mode == "css3":
        names, rgbs, labs = _base_names, _base_rgbs, _base_labs
    else:
        names, rgbs, labs = _build_css3x2(delta_L=delta_L)

    # candidate lab
    lab = _rgb_to_lab(np.array(rgb_uint8_triplet, dtype=np.uint8)[None,:])  # shape (1,3)
    if use_ciede2000:
        d = _delta_e_ciede2000(labs, lab)
        idx = int(np.argmin(d))
    else:
        d2 = np.sum((labs - lab)**2, axis=1)
        idx = int(np.argmin(d2))
    return names[idx]

def _unique_rows(a):
    a = np.ascontiguousarray(a)
    return np.unique(a.view([('', a.dtype)]*a.shape[1])).view(a.dtype).reshape(-1, a.shape[1])

def _kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=np.float32)
    i0 = int(rng.integers(0, n))
    centroids[0] = X[i0]
    closest = np.sum((X - centroids[0])**2, axis=1)
    for c in range(1, k):
        s = float(closest.sum())
        if not np.isfinite(s) or s <= 1e-12:
            idx = int(rng.integers(0, n))
        else:
            probs = closest / max(s, 1e-12)
            probs = np.clip(probs, 0.0, 1.0)
            z = probs.sum()
            if z <= 0 or not np.isfinite(z):
                idx = int(rng.integers(0, n))
            else:
                probs = probs / z
                idx = int(rng.choice(n, p=probs))
        centroids[c] = X[idx]
        dist_sq = np.sum((X - centroids[c])**2, axis=1)
        closest = np.minimum(closest, dist_sq)
    return centroids

def _kmeans(X, k=10, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    centroids = _kmeans_pp_init(X, k, rng)
    for _ in range(iters):
        d = np.sum((X[:,None,:] - centroids[None,:,:])**2, axis=2)
        labels = np.argmin(d, axis=1)
        for j in range(k):
            m = labels == j
            if np.any(m): centroids[j] = X[m].mean(axis=0)
            else:         centroids[j] = X[int(rng.integers(0, X.shape[0]))]
    d = np.sum((X[:,None,:] - centroids[None,:,:])**2, axis=2)
    labels = np.argmin(d, axis=1)
    return centroids, labels

def _circ_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def _nearest_by_hue(target_h, palette, exclude_indices=set()):
    best = None
    best_idx = -1
    best_d = 1e9
    for i, it in enumerate(palette):
        if i in exclude_indices:
            continue
        d = _circ_dist(target_h, it["h"])
        if d < best_d:
            best_d = d; best = it; best_idx = i
    return best_idx, best

class TwoStrongestColors:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                # keep legacy order first
                "alpha_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ignore_near_white": ("BOOLEAN", {"default": True}),
                "ignore_near_black": ("BOOLEAN", {"default": True}),
                "whiteness_threshold": ("FLOAT", {"default": 0.92, "min": 0.5, "max": 1.0, "step": 0.01}),
                "blackness_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "k_clusters": ("INT", {"default": 10, "min": 3, "max": 16}),
                "sample_limit": ("INT", {"default": 200000, "min": 1000, "max": 2000000}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
                "top_n": ("INT", {"default": 10, "min": 3, "max": 16}),
                "percent_round": ("INT", {"default": 2, "min": 0, "max": 6}),
                "emit_markdown": ("BOOLEAN", {"default": True}),
                # new params appended (no widget reorder)
                "bank_mode": (["css3", "css3x2"], {"default": "css3x2"}),
                "expand_delta_L": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 30.0, "step": 1.0}),
                "use_ciede2000": ("BOOLEAN", {"default": True}),
            },
        }

    # Outputs: back-compat + palette JSON & text
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("color_1_name", "color_2_name", "palette_json", "palette_text")
    FUNCTION = "analyze"
    CATEGORY = "Color/Analysis"

    def _prepare_pixels(self, img_tensor, alpha_threshold, ignore_near_white, ignore_near_black,
                        whiteness_threshold, blackness_threshold, sample_limit, seed):
        if img_tensor.dim() != 4:
            raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")

        B,H,W,C = img_tensor.shape
        if C < 3:
            raise ValueError("Image must have at least 3 channels (RGB).")
        if C > 4:
            img_tensor = img_tensor[...,:4]; C = 4

        img = np.clip(img_tensor.detach().cpu().numpy(), 0.0, 1.0)

        # Transparent-only early exit
        if C == 4:
            a = img[...,3]
            if np.all(a <= alpha_threshold):
                return np.zeros((0,3), dtype=np.uint8), True

        if C == 4:
            rgb = (img[...,:3]*255.0).astype(np.uint8).reshape(-1,3)
            a   = img[...,3].reshape(-1)
            alpha_mask = a > alpha_threshold
        else:
            rgb = (img[...,:3]*255.0).astype(np.uint8).reshape(-1,3)
            alpha_mask = np.ones((rgb.shape[0],), dtype=bool)

        rgb_norm = rgb.astype(np.float32)/255.0
        intensity = rgb_norm.mean(axis=1)

        mask = alpha_mask.copy()
        if ignore_near_white: mask &= (intensity < whiteness_threshold)
        if ignore_near_black: mask &= (intensity > blackness_threshold)

        rgb = rgb[mask]
        if rgb.size == 0:
            return rgb, False

        if rgb.shape[0] > sample_limit:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, rgb.shape[0], size=sample_limit)
            rgb = rgb[idx]
        return rgb, False

    def analyze(
        self, image,
        alpha_threshold=0.05, ignore_near_white=True, ignore_near_black=True,
        whiteness_threshold=0.92, blackness_threshold=0.08,
        k_clusters=10, sample_limit=200000, random_seed=0,
        top_n=10, percent_round=2, emit_markdown=True,
        bank_mode="css3x2", expand_delta_L=12.0, use_ciede2000=True
    ):
        # --- Back-compat / migration guard ---
        if top_n > 64 and sample_limit < 1000:
            sample_limit = int(max(1000, top_n))
            top_n = 10

        # Clamp safe bounds
        sample_limit = int(max(1000, sample_limit))
        k_clusters = int(max(3, min(16, k_clusters)))
        top_n = int(max(3, min(16, top_n)))
        expand_delta_L = float(min(30.0, max(2.0, expand_delta_L)))

        # Accept single tensor or list
        tensors = []
        if isinstance(image, list):
            tensors = [t for t in image if torch.is_tensor(t)]
        elif torch.is_tensor(image):
            tensors = [image]
        if not tensors:
            return (" ", " ", " ", " ")

        try:
            img_tensor = torch.cat(tensors, dim=0)
        except Exception:
            img_tensor = tensors[0]

        # Prepare pixels
        pixels, was_all_transparent = self._prepare_pixels(
            img_tensor, alpha_threshold, ignore_near_white, ignore_near_black,
            whiteness_threshold, blackness_threshold, sample_limit, random_seed
        )

        if was_all_transparent:
            return (" ", " ", " ", " ")

        if pixels.size == 0:
            pixels, _ = self._prepare_pixels(
                img_tensor, 0.0, False, False, 1.0, 0.0, sample_limit, random_seed
            )
            if pixels.size == 0:
                return (" ", " ", " ", " ")

        # Unique cap
        uniq = _unique_rows(pixels.astype(np.float32))
        n_unique = uniq.shape[0]
        k = int(max(3, min(k_clusters, n_unique)))
        top_n = int(max(3, min(top_n, k)))

        if n_unique == 1:
            rgb = np.clip(np.round(uniq[0]), 0, 255).astype(np.uint8)
            name = _nearest_named_color(rgb, bank_mode=bank_mode, use_ciede2000=use_ciede2000, delta_L=expand_delta_L)
            hexv = _rgb_to_hex(rgb)
            base_item = {
                "name": name, "hex": hexv, "rgb": rgb.tolist(),
                "lab": _rgb_to_lab(rgb[None,:])[0].round(3).tolist(),
                "hsv": list(map(lambda x: round(float(x), 3), _rgb_to_hsv_deg(rgb))),
                "observed_percent": 100.0, "role": "dominant"
            }
            palette = {
                "rule": "60-30-10",
                "dominant": {**base_item, "assigned_percent": 60},
                "secondary": {**base_item, "assigned_percent": 30},
                "accent": {**base_item, "assigned_percent": 10},
                "top_colors": [base_item],
                "harmonies": {"analogous": [], "triadic": [], "complementary": [], "monochromatic": []}
            }
            palette_json = json.dumps(palette, ensure_ascii=False)
            md = (
                f"**60-30-10 Palette**\n"
                f"- 60% Dominant: {name} ({hexv})\n"
                f"- 30% Secondary: {name} ({hexv})\n"
                f"- 10% Accent: {name} ({hexv})\n"
                f"\nTop colors (1): 100% {name} ({hexv})"
            )
            return (name, name, palette_json, md if emit_markdown else " ")

        # K-means on RGB
        X = pixels.astype(np.float32)
        centroids, labels = _kmeans(X, k=k, iters=15, seed=random_seed)
        counts = np.bincount(labels, minlength=centroids.shape[0])
        order = np.argsort(-counts)

        total = float(counts.sum())
        items = []
        for idx in order[:top_n]:
            rgb = np.clip(np.round(centroids[idx]), 0, 255).astype(np.uint8)
            name = _nearest_named_color(rgb, bank_mode=bank_mode, use_ciede2000=use_ciede2000, delta_L=expand_delta_L)
            hexv = _rgb_to_hex(rgb)
            L,a,b = _rgb_to_lab(rgb[None,:])[0]
            h,s,v = _rgb_to_hsv_deg(rgb)
            items.append({
                "name": name,
                "hex": hexv,
                "rgb": rgb.tolist(),
                "lab": [float(round(L,3)), float(round(a,3)), float(round(b,3))],
                "hsv": [float(round(h,3)), float(round(s,3)), float(round(v,3))],
                "observed_percent": float(round((counts[idx]/total)*100.0, percent_round)),
                "role": "supporting"
            })

        # Assign 60/30/10
        if len(items) >= 1: items[0]["role"] = "dominant"
        if len(items) >= 2: items[1]["role"] = "secondary"
        if len(items) >= 3: items[2]["role"] = "accent"

        # Harmony suggestions (by hue)
        dominant = items[0]
        dom_h = dominant["hsv"][0]
        used = {0}

        def _hue_list(its): return [{"h": it["hsv"][0]} for it in its]

        analogous = []
        for t in [(dom_h + 30.0) % 360.0, (dom_h - 30.0) % 360.0]:
            idx, _ = _nearest_by_hue(t, _hue_list(items), exclude_indices=used)
            if idx >= 0:
                used.add(idx); analogous.append(items[idx])

        triadic = []
        for t in [(dom_h + 120.0) % 360.0, (dom_h - 120.0) % 360.0]:
            idx, _ = _nearest_by_hue(t, _hue_list(items), exclude_indices=used)
            if idx >= 0:
                used.add(idx); triadic.append(items[idx])

        comp_idx, _ = _nearest_by_hue((dom_h + 180.0) % 360.0, _hue_list(items), exclude_indices=set([0]))
        complementary = [items[comp_idx]] if comp_idx >= 0 else []

        mono = [it for it in items if _circ_dist(dom_h, it["hsv"][0]) <= 12.0]
        mono = sorted(mono, key=lambda it: it["hsv"][2], reverse=True)

        palette = {
            "rule": "60-30-10",
            "dominant": {**items[0], "assigned_percent": 60} if len(items) >= 1 else None,
            "secondary": {**items[1], "assigned_percent": 30} if len(items) >= 2 else None,
            "accent": {**items[2], "assigned_percent": 10} if len(items) >= 3 else None,
            "top_colors": items,
            "harmonies": {
                "analogous": analogous,
                "triadic": triadic,
                "complementary": complementary,
                "monochromatic": mono
            }
        }

        palette_json = json.dumps(palette, ensure_ascii=False)

        strongest_1 = items[0]["name"] if len(items) >= 1 else " "
        strongest_2 = items[1]["name"] if len(items) >= 2 else strongest_1

        if emit_markdown:
            lines = []
            lines.append("**60-30-10 Palette**")
            if len(items) >= 1:
                lines.append(f"- 60% Dominant: {items[0]['name']} ({items[0]['hex']})")
            if len(items) >= 2:
                lines.append(f"- 30% Secondary: {items[1]['name']} ({items[1]['hex']})")
            if len(items) >= 3:
                lines.append(f"- 10% Accent: {items[2]['name']} ({items[2]['hex']})")
            lines.append("")
            lines.append(f"**Top {len(items)} Colors (observed)**")
            for i,it in enumerate(items, 1):
                lines.append(f"{i}. {it['observed_percent']}% – {it['name']} ({it['hex']})")
            lines.append("")
            def _fmt_group(title, group):
                if not group: return f"**{title}:** (none)"
                return "**" + title + ":** " + ", ".join([f"{x['name']} ({x['hex']})" for x in group])
            lines.append(_fmt_group("Analogous", analogous))
            lines.append(_fmt_group("Triadic", triadic))
            lines.append(_fmt_group("Complementary", complementary))
            lines.append(_fmt_group("Monochromatic", mono))
            palette_text = "\n".join(lines)
        else:
            palette_text = " "

        return (strongest_1, strongest_2, palette_json, palette_text)


NODE_CLASS_MAPPINGS = {"TwoStrongestColors": TwoStrongestColors}
NODE_DISPLAY_NAME_MAPPINGS = {"TwoStrongestColors": "Two Strongest Colors + Palette (css3x2 + ΔE00)"}
