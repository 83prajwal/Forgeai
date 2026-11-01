/* Tiny offline QR encoder: byte mode, QR version 4-L, local to ForgeAI. */
(() => {
  const SIZE = 33;
  const DATA_CODEWORDS = 80;
  const ECC_CODEWORDS = 20;
  const gfMultiply = (x, y) => {
    let z = 0;
    for (let i = 7; i >= 0; i -= 1) {
      z = (z << 1) ^ ((z >>> 7) * 0x11d);
      z ^= ((y >>> i) & 1) * x;
    }
    return z;
  };

  function appendBits(bits, value, count) {
    for (let i = count - 1; i >= 0; i -= 1) bits.push((value >>> i) & 1);
  }

  function makeCodewords(text) {
    const bytes = [...new TextEncoder().encode(text)];
    if (bytes.length > 78) throw new Error('Preview link is too long for this offline QR encoder.');
    const bits = [];
    appendBits(bits, 0x4, 4);
    appendBits(bits, bytes.length, 8);
    bytes.forEach((byte) => appendBits(bits, byte, 8));
    const capacity = DATA_CODEWORDS * 8;
    appendBits(bits, 0, Math.min(4, capacity - bits.length));
    while (bits.length % 8) bits.push(0);
    const data = [];
    for (let i = 0; i < bits.length; i += 8) {
      data.push(bits.slice(i, i + 8).reduce((value, bit) => (value << 1) | bit, 0));
    }
    for (let pad = 0; data.length < DATA_CODEWORDS; pad += 1) data.push(pad % 2 === 0 ? 0xec : 0x11);

    let divisor = [1];
    let root = 1;
    for (let i = 0; i < ECC_CODEWORDS; i += 1) {
      const next = new Array(divisor.length + 1).fill(0);
      divisor.forEach((coefficient, index) => {
        next[index] ^= coefficient;
        next[index + 1] ^= gfMultiply(coefficient, root);
      });
      divisor = next;
      root = gfMultiply(root, 2);
    }
    const remainder = [...data, ...new Array(ECC_CODEWORDS).fill(0)];
    for (let i = 0; i < data.length; i += 1) {
      const factor = remainder[i];
      if (factor) divisor.forEach((coefficient, j) => { remainder[i + j] ^= gfMultiply(coefficient, factor); });
    }
    return [...data, ...remainder.slice(data.length)];
  }

  function formatBits(mask) {
    const data = (1 << 3) | mask; // Error correction level L.
    let remainder = data << 10;
    for (let bit = 14; bit >= 10; bit -= 1) {
      if ((remainder >>> bit) & 1) remainder ^= 0x537 << (bit - 10);
    }
    return ((data << 10) | remainder) ^ 0x5412;
  }

  function buildMatrix(text) {
    const matrix = Array.from({ length: SIZE }, () => new Array(SIZE).fill(null));
    const setFunction = (x, y, dark) => { if (x >= 0 && y >= 0 && x < SIZE && y < SIZE) matrix[y][x] = Boolean(dark); };
    const finder = (cx, cy) => {
      for (let dy = -4; dy <= 4; dy += 1) {
        for (let dx = -4; dx <= 4; dx += 1) {
          const distance = Math.max(Math.abs(dx), Math.abs(dy));
          setFunction(cx + dx, cy + dy, distance === 3 || distance <= 1);
        }
      }
    };
    finder(3, 3); finder(SIZE - 4, 3); finder(3, SIZE - 4);
    for (let i = 8; i < SIZE - 8; i += 1) {
      if (matrix[6][i] === null) setFunction(i, 6, i % 2 === 0);
      if (matrix[i][6] === null) setFunction(6, i, i % 2 === 0);
    }
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) setFunction(26 + dx, 26 + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
    }
    // Reserve the format-information modules before writing data.
    for (let i = 0; i <= 5; i += 1) setFunction(8, i, false);
    setFunction(8, 7, false); setFunction(8, 8, false); setFunction(7, 8, false);
    for (let i = 9; i < 15; i += 1) setFunction(14 - i, 8, false);
    for (let i = 0; i < 8; i += 1) setFunction(SIZE - 1 - i, 8, false);
    for (let i = 8; i < 15; i += 1) setFunction(8, SIZE - 15 + i, false);
    setFunction(8, SIZE - 8, true);

    const codewords = makeCodewords(text);
    const bits = codewords.flatMap((word) => Array.from({ length: 8 }, (_, i) => (word >>> (7 - i)) & 1));
    let bitIndex = 0;
    let upward = true;
    for (let right = SIZE - 1; right >= 1; right -= 2) {
      if (right === 6) right -= 1;
      for (let vert = 0; vert < SIZE; vert += 1) {
        const y = upward ? SIZE - 1 - vert : vert;
        for (let offset = 0; offset < 2; offset += 1) {
          const x = right - offset;
          if (matrix[y][x] !== null) continue;
          const raw = bitIndex < bits.length ? bits[bitIndex] : 0;
          bitIndex += 1;
          matrix[y][x] = Boolean(raw ^ ((x + y) % 2 === 0 ? 1 : 0)); // Mask 0.
        }
      }
      upward = !upward;
    }
    const format = formatBits(0);
    for (let i = 0; i <= 5; i += 1) setFunction(8, i, ((format >>> i) & 1) !== 0);
    setFunction(8, 7, ((format >>> 6) & 1) !== 0);
    setFunction(8, 8, ((format >>> 7) & 1) !== 0);
    setFunction(7, 8, ((format >>> 8) & 1) !== 0);
    for (let i = 9; i < 15; i += 1) setFunction(14 - i, 8, ((format >>> i) & 1) !== 0);
    for (let i = 0; i < 8; i += 1) setFunction(SIZE - 1 - i, 8, ((format >>> i) & 1) !== 0);
    for (let i = 8; i < 15; i += 1) setFunction(8, SIZE - 15 + i, ((format >>> i) & 1) !== 0);
    setFunction(8, SIZE - 8, true);
    return matrix;
  }

  function toSvg(text) {
    const matrix = buildMatrix(text);
    const quiet = 4;
    const dimension = SIZE + quiet * 2;
    let path = '';
    matrix.forEach((row, y) => row.forEach((dark, x) => {
      if (dark) path += `M${x + quiet},${y + quiet}h1v1h-1z`;
    }));
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dimension} ${dimension}" shape-rendering="crispEdges"><rect width="100%" height="100%" fill="#fff"/><path d="${path}" fill="#111"/></svg>`;
  }

  window.ForgeQR = { toSvg };
})();
