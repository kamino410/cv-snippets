let gl = null;
let canvas = null;
let state = {
  pro1: null,
  vpbo: null,
  ibo: null,
  tbo1: null,
  tbo2: null,
  iboLength: null,
  tex: null,
  fx: null,
  fy: null,
  cx: null,
  cy: null,
  k1: null,
  k2: null,
  k3: null,
  k4: null,
  k5: null,
  k6: null,
  p1: null,
  p2: null,
  s1: null,
  s2: null,
  s3: null,
  s4: null,
};

const vertSrc1 = `
attribute vec3 position;

varying vec2 colpos;
uniform float w;
uniform float h;
uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;
uniform float k1;
uniform float k2;
uniform float k3;
uniform float k4;
uniform float k5;
uniform float k6;
uniform float p1;
uniform float p2;
uniform float s1;
uniform float s2;
uniform float s3;
uniform float s4;

void main() {
  float x = position.x;
  float y = position.y;
  float z = position.z;
  float r2 = x*x + y*y;
  float r4 = r2*r2;
  float r6 = r4*r2;

  float c = (1.0 + k1*r2 + k2*r4 + k3*r6)/(1.0 + k4*r2 + k5*r4 + k6*r6) + 2.0*p1*y + 2.0*p2*x;
  if (c < 0.0) z = 1.0; // not drawn

  float tx = p2*r2 + s1*r2 + s2*r4;
  float ty = p1*r2 + s3*r2 + s4*r4;

  float xd = x*c + tx;
  float yd = y*c + ty;
  colpos = 0.5*vec2(x, y)+vec2(0.5,0.5);

  gl_Position = vec4(2.0*(fx*xd+cx)/w-1.0, -2.0*(fy*yd+cy)/h+1.0, z, 1.0);
}
`;

const fragSrc1 = `
precision highp float;

varying vec2 colpos;
uniform sampler2D tex1;

void main() {
  gl_FragColor = texture2D(tex1, colpos);
}
`;

function initShader() {
  var vertShader1 = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertShader1, vertSrc1);
  gl.compileShader(vertShader1);
  if (!gl.getShaderParameter(vertShader1, gl.COMPILE_STATUS)) {
    console.log(gl.getShaderInfoLog(vertShader1));
  }

  var fragShader1 = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragShader1, fragSrc1);
  gl.compileShader(fragShader1);
  if (!gl.getShaderParameter(fragShader1, gl.COMPILE_STATUS)) {
    console.log(gl.getShaderInfoLog(fragShader1));
  }

  var pro1 = gl.createProgram();
  gl.attachShader(pro1, vertShader1);
  gl.attachShader(pro1, fragShader1);
  gl.linkProgram(pro1);
  if (!gl.getProgramParameter(pro1, gl.LINK_STATUS)) {
    console.log('failed to link program');
  }
  gl.useProgram(pro1);
  state.pro1 = pro1;
}

function setupScene() {
  const vertexPosition = [];
  for (var x = -15; x <= 15; x++) {
    for (var y = -15; y <= 15; y++) {
      vertexPosition.push(x / 10);
      vertexPosition.push(y / 10);
      vertexPosition.push(0.01 * (Math.abs(x) + Math.abs(y)) + 0.01);
    }
  }

  const indexes = [];
  for (var x = 0; x < 30; x++) {
    for (var y = 0; y < 30; y++) {
      indexes.push(31 * y + x);
      indexes.push(31 * y + x + 1);
      indexes.push(31 * (y + 1) + x);

      indexes.push(31 * y + x + 1);
      indexes.push(31 * (y + 1) + x);
      indexes.push(31 * (y + 1) + x + 1);
    }
  }
  state.iboLength = indexes.length;

  var vpbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vpbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexPosition), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  state.vpbo = vpbo;

  var posAttrLoc = gl.getAttribLocation(state.pro1, 'position');
  gl.bindBuffer(gl.ARRAY_BUFFER, state.vpbo);
  gl.enableVertexAttribArray(posAttrLoc);
  gl.vertexAttribPointer(posAttrLoc, 3, gl.FLOAT, false, 0, 0);

  var ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(
    gl.ELEMENT_ARRAY_BUFFER, new Int16Array(indexes), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  state.ibo = ibo;

  var tbo1 = gl.createTexture();
  var img1 = new Image();
  img1.onload = function () {
    gl.bindTexture(gl.TEXTURE_2D, tbo1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  img1.src = './grid.png';
  state.tbo1 = tbo1;

  var tbo2 = gl.createTexture();
  var img2 = new Image();
  img2.onload = function () {
    gl.bindTexture(gl.TEXTURE_2D, tbo2);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img2);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  img2.src = './buildings.png';
  state.tbo2 = tbo2;

  state.tex = 0;

  gl.enable(gl.DEPTH_TEST);
}

function drawScene(time) {
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(state.pro1);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.tbo1);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.tbo2);
  gl.uniform1i(gl.getUniformLocation(state.pro1, 'tex1'), state.tex);

  gl.uniform1f(gl.getUniformLocation(state.pro1, 'w'), canvas.width);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'h'), canvas.height);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'fx'), state.fx);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'fy'), state.fy);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'cx'), state.cx);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'cy'), state.cy);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'k1'), state.k1);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'k2'), state.k2);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'k3'), state.k3);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'k4'), state.k4);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'k5'), state.k5);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'k6'), state.k6);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'p1'), state.p1);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 'p2'), state.p2);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 's1'), state.s1);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 's2'), state.s2);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 's3'), state.s3);
  gl.uniform1f(gl.getUniformLocation(state.pro1, 's4'), state.s4);

  gl.bindBuffer(gl.ARRAY_BUFFER, state.vpbo);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.ibo);
  gl.drawElements(gl.TRIANGLES, state.iboLength, gl.UNSIGNED_SHORT, 0);

  requestAnimationFrame(drawScene);
}

function main() {
  canvas = document.getElementById('gl-canvas');
  canvas.width = 480;
  canvas.height = 360;

  gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

  initShader();
  setupScene();
  requestAnimationFrame(drawScene)

  const imgsrc = document.getElementById('tex-src');
  imgsrc.onchange = function () {
    state.tex = this.options[this.selectedIndex].value;
  };

  const fx = document.getElementById('fx');
  const labelfx = document.getElementById('lab-fx');
  labelfx.innerHTML = state.fx = fx.value - 0;
  fx.oninput = function () { labelfx.innerHTML = state.fx = fx.value - 0; };

  const cx = document.getElementById('cx');
  const labelcx = document.getElementById('lab-cx');
  labelcx.innerHTML = state.cx = cx.value - 0;
  cx.oninput = function () { labelcx.innerHTML = state.cx = cx.value - 0; };

  const fy = document.getElementById('fy');
  const labelfy = document.getElementById('lab-fy');
  labelfy.innerHTML = state.fy = fy.value - 0;
  fy.oninput = function () { labelfy.innerHTML = state.fy = fy.value - 0; };

  const cy = document.getElementById('cy');
  const labelcy = document.getElementById('lab-cy');
  labelcy.innerHTML = state.cy = cy.value - 0;
  cy.oninput = function () { labelcy.innerHTML = state.cy = cy.value - 0; };

  const k1 = document.getElementById('k1');
  const labelk1 = document.getElementById('lab-k1');
  labelk1.innerHTML = state.k1 = k1.value - 0;
  k1.oninput = function () { labelk1.innerHTML = state.k1 = k1.value - 0; };

  const k2 = document.getElementById('k2');
  const labelk2 = document.getElementById('lab-k2');
  labelk2.innerHTML = state.k2 = k2.value - 0;
  k2.oninput = function () { labelk2.innerHTML = state.k2 = k2.value - 0; };

  const k3 = document.getElementById('k3');
  const labelk3 = document.getElementById('lab-k3');
  labelk3.innerHTML = state.k3 = k3.value - 0;
  k3.oninput = function () { labelk3.innerHTML = state.k3 = k3.value - 0; };

  const k4 = document.getElementById('k4');
  const labelk4 = document.getElementById('lab-k4');
  labelk4.innerHTML = state.k4 = k4.value - 0;
  k4.oninput = function () { labelk4.innerHTML = state.k4 = k4.value - 0; };

  const k5 = document.getElementById('k5');
  const labelk5 = document.getElementById('lab-k5');
  labelk5.innerHTML = state.k5 = k5.value - 0;
  k5.oninput = function () { labelk5.innerHTML = state.k5 = k5.value - 0; };

  const k6 = document.getElementById('k6');
  const labelk6 = document.getElementById('lab-k6');
  labelk6.innerHTML = state.k6 = k6.value - 0;
  k6.oninput = function () { labelk6.innerHTML = state.k6 = k6.value - 0; };

  const p1 = document.getElementById('p1');
  const labelp1 = document.getElementById('lab-p1');
  labelp1.innerHTML = state.p1 = p1.value - 0;
  p1.oninput = function () { labelp1.innerHTML = state.p1 = p1.value - 0; };

  const p2 = document.getElementById('p2');
  const labelp2 = document.getElementById('lab-p2');
  labelp2.innerHTML = state.p2 = p2.value - 0;
  p2.oninput = function () { labelp2.innerHTML = state.p2 = p2.value - 0; };

  const s1 = document.getElementById('s1');
  const labels1 = document.getElementById('lab-s1');
  labels1.innerHTML = state.s1 = s1.value - 0;
  s1.oninput = function () { labels1.innerHTML = state.s1 = s1.value - 0; };

  const s2 = document.getElementById('s2');
  const labels2 = document.getElementById('lab-s2');
  labels2.innerHTML = state.s2 = s2.value - 0;
  s2.oninput = function () { labels2.innerHTML = state.s2 = s2.value - 0; };

  const s3 = document.getElementById('s3');
  const labels3 = document.getElementById('lab-s3');
  labels3.innerHTML = state.s3 = s3.value - 0;
  s3.oninput = function () { labels3.innerHTML = state.s3 = s3.value - 0; };

  const s4 = document.getElementById('s4');
  const labels4 = document.getElementById('lab-s4');
  labels4.innerHTML = state.s4 = s4.value - 0;
  s4.oninput = function () { labels4.innerHTML = state.s4 = s4.value - 0; };

  const reset = document.getElementById('reset');
  reset.onclick = function () {
    fx.value = 480; fx.oninput.call();
    fy.value = 480; fy.oninput.call();
    cx.value = 240; cx.oninput.call();
    cy.value = 180; cy.oninput.call();
    k1.value = 0; k1.oninput.call();
    k2.value = 0; k2.oninput.call();
    k3.value = 0; k3.oninput.call();
    k4.value = 0; k4.oninput.call();
    k5.value = 0; k5.oninput.call();
    k6.value = 0; k6.oninput.call();
    p1.value = 0; p1.oninput.call();
    p2.value = 0; p2.oninput.call();
    s1.value = 0; s1.oninput.call();
    s2.value = 0; s2.oninput.call();
    s3.value = 0; s3.oninput.call();
    s4.value = 0; s4.oninput.call();
  };
}

onload = main;
