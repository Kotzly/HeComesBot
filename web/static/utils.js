// ── Tree lookup ───────────────────────────────────────────────────────────────
export function findNode(nodes, id) {
  return (nodes && nodes[id]) || null;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────
export function jsonHdr() { return {'Content-Type': 'application/json'}; }

// ── Color conversions ─────────────────────────────────────────────────────────
export function hexToRgb(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

export function rgbToHex(r, g, b) {
  const h = v => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0');
  return `#${h(r)}${h(g)}${h(b)}`;
}

export function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p]; case 1: return [q, v, p]; case 2: return [p, v, t];
    case 3: return [p, q, v]; case 4: return [t, p, v]; case 5: return [v, p, q];
  }
}

export function rgbToHsv(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  const s = max === 0 ? 0 : d / max, v = max;
  let h = 0;
  if (d !== 0) {
    if      (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else                h = ((r - g) / d + 4) / 6;
  }
  return [h, s, v];
}

export function getColorSpace() {
  return (document.getElementById('color-space-select') || {}).value || 'rgb';
}

export function colorParamToHex(c) {
  const cs = getColorSpace();
  if (cs === 'hsv') {
    const [r, g, b] = hsvToRgb(((c[0] % 1) + 1) % 1, Math.min(1, Math.max(0, c[1])), Math.min(1, Math.max(0, c[2])));
    return rgbToHex(r, g, b);
  }
  if (cs === 'cmy') return rgbToHex(1 - c[0], 1 - c[1], 1 - c[2]);
  return rgbToHex(c[0], c[1], c[2]);
}

export function hexToColorParam(hex) {
  const [r, g, b] = hexToRgb(hex);
  const cs = getColorSpace();
  if (cs === 'hsv') return rgbToHsv(r, g, b);
  if (cs === 'cmy') return [1 - r, 1 - g, 1 - b];
  return [r, g, b];
}
