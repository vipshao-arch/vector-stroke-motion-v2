import * as THREE from './three.module.js?v=8';
import {
  KEYFRAMES, KEYFRAME_FIELDS, parseInstructions,
} from './instructions.js?v=8';
import { DynamicOverlay } from './dynamicOverlay.js?v=9';

const VERTEX_SHADER = /* glsl */ `
  precision highp float;
  attribute vec3 position;
  attribute float aIndex;
  uniform sampler2D uParams;
  uniform sampler2D uColor;
  uniform sampler2D uMotion;
  uniform sampler2D uKeys;
  uniform vec2 uParamSize;
  uniform vec2 uColorSize;
  uniform vec2 uMotionSize;
  uniform vec2 uKeySize;
  uniform float uTime;
  uniform float uFrame;
  uniform vec2 uLogicalSize;
  varying vec2 vLocal;
  varying vec2 vEllipseLocal;
  varying vec2 vStart;
  varying vec2 vControl;
  varying vec2 vEnd;
  varying vec2 vSize;
  varying float vWidth;
  varying float vPetalBend;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vKind;

  vec4 texel(sampler2D tex, vec2 size, float offset) {
    float width = max(size.x, 1.0);
    float height = max(size.y, 1.0);
    float index = clamp(offset, 0.0, width * height - 1.0);
    float x = mod(index, width);
    float y = floor(index / width);
    // Callers derive x/y from integer texel indices. The half-texel guard
    // avoids precision-dependent nearest sampling on large float textures.
    return texture2D(tex, (floor(vec2(x, y)) + 0.5) / size);
  }

  float readKey(float stroke, float field, float key) {
    float offset = (stroke * ${KEYFRAMES}.0 + key) * ${KEYFRAME_FIELDS}.0 + field;
    float texelIndex = floor(offset * 0.25);
    vec4 value = texel(uKeys, uKeySize, texelIndex);
    float component = mod(offset, 4.0);
    if (component < 0.5) return value.r;
    if (component < 1.5) return value.g;
    if (component < 2.5) return value.b;
    return value.a;
  }

  float interpolateKey(float stroke, float field, float position) {
    float key = floor(position);
    float blend = smoothstep(0.0, 1.0, position - key);
    float a = readKey(stroke, field, key);
    float b = readKey(stroke, field, min(key + 1.0, ${KEYFRAMES - 1}.0));
    return mix(a, b, blend);
  }

  void main() {
    float stroke = aIndex;
    vec4 block = texel(uParams, uParamSize, stroke * 4.0);
    // Kind and motion model are packed as normalized bytes in the first vec4.
    float kind = block.r * 255.0;
    float model = block.b * 255.0;
    float alphaBase = block.a;
    vec4 p0 = texel(uParams, uParamSize, stroke * 4.0 + 1.0);
    vec4 p1 = texel(uParams, uParamSize, stroke * 4.0 + 2.0);
    vec4 ink = texel(uColor, uColorSize, stroke);
    vec4 m0 = texel(uMotion, uMotionSize, stroke * 2.0);
    vec4 m1 = texel(uMotion, uMotionSize, stroke * 2.0 + 1.0);
    float keyPosition = clamp(uTime, 0.0, 1.0) * ${KEYFRAMES - 1}.0;
    float dx = interpolateKey(stroke, 0.0, keyPosition);
    float dy = interpolateKey(stroke, 1.0, keyPosition);
    float scale = interpolateKey(stroke, 2.0, keyPosition);
    float rotate = interpolateKey(stroke, 3.0, keyPosition);
    float deltaAlpha = interpolateKey(stroke, 4.0, keyPosition);
    float brightness = interpolateKey(stroke, 5.0, keyPosition);
    float wobble = interpolateKey(stroke, 7.0, keyPosition);
    float phase = interpolateKey(stroke, 6.0, keyPosition);
    brightness += abs(phase - keyPosition / ${KEYFRAMES - 1}.0) * 0.000001;

    vec2 origin;
    vec2 size;
    vec2 control;
    vec2 end;
    vec2 start;
    float rotation;
    if (kind < 0.5) {
      origin = p0.xy;
      size = p0.zw;
      rotation = p1.x;
      control = vec2(0.0);
      end = vec2(0.0);
      start = vec2(0.0);
    } else if (kind > 0.5 && kind < 3.5) {
      vec2 a = p0.xy;
      vec2 b = p0.zw;
      vec2 c = p1.xy;
      origin = (a + b + c) / 3.0;
      float reach = max(max(length(a - origin), length(b - origin)), length(c - origin));
      size = vec2(reach + p1.z * 2.0, reach + p1.z * 2.0);
      control = b - origin;
      end = c - origin;
      start = a - origin;
      rotation = 0.0;
      // Hair phase propagation and wing bending move the control point along
      // the local curve normal. Each stroke has its own phase and gain in m0/m1.
      if (model > 1.5 && model < 4.5) {
        float bendAmplitude = mix(m0.z, m0.z * 0.45, step(2.5, model));
        vec2 tangent = normalize(c - a + vec2(0.0001, 0.0));
        vec2 normal = vec2(-tangent.y, tangent.x);
        float bend = bendAmplitude * sin(uFrame * m0.y + m0.x + m0.w * length(b - a));
        control += normal * bend;
        start += normal * bend * 0.35;
        end += normal * bend * 0.8;
      }
    } else if (kind < 4.5) {
      // Petal or shimmer: a rotated, elongated soft ellipse.
      origin = p0.xy;
      size = p0.zw;
      control = vec2(0.0);
      end = vec2(0.0);
      start = vec2(0.0);
      rotation = p1.x;
    } else if (kind < 5.5) {
      // Soft residual brush: round Gaussian-like cap at p0.xy.
      origin = p0.xy;
      size = vec2(p0.z, p0.z);
      control = vec2(0.0);
      end = vec2(0.0);
      start = vec2(0.0);
      rotation = 0.0;
    } else {
      // Shimmer also uses a compact soft circle at p0.xy with radius p0.z.
      origin = p0.xy;
      size = vec2(p0.z, p0.z);
      control = vec2(0.0);
      end = vec2(0.0);
      start = vec2(0.0);
      rotation = 0.0;
    }

    // The breathing model is centered on the chest. All other static regions
    // keep scale=1, rotate=0 and no displacement through their keyframes.
    if (model > 0.5 && model < 1.5) {
      vec2 chest = m0.xy;
      vec2 radial = origin - chest;
      float falloff = exp(-dot(radial, radial) / max(m1.x * m1.x, 1.0));
      float breath = sin(uFrame * m0.y + m0.x);
      origin += radial * (0.009 * breath * falloff);
      scale *= 1.0 + 0.008 * breath * falloff;
      brightness *= 1.0 + 0.035 * breath * falloff;
    }

    if (kind > 3.5 && kind < 4.5) {
      // Parametric petal life: spawn time, downward speed, horizontal sway,
      // spin and fade envelope are all stored per stroke.
      float life = fract(uTime * m1.x + m0.z);
      float fall = uFrame * m0.y;
      float sway = sin(uFrame * m1.y + m0.x) * m1.z;
      float wrap = floor(max(fall - m0.w + ${'352.0'}, 0.0) / ${'520.0'}) * ${'520.0'};
      origin += vec2(sway, fall - m0.w - wrap);
      rotate += m1.w * life + sin(uFrame * m1.y * 0.5 + m0.x) * 0.35;
      float envelope = smoothstep(0.0, 0.09, life) * (1.0 - smoothstep(0.72, 1.0, life));
      alphaBase *= envelope;
    }

    vKind = kind;
    vSize = size;
    vColor = clamp(ink.rgb * brightness, 0.0, 1.0);
    vColor.r += abs(phase - keyPosition / ${KEYFRAMES - 1}.0) * 0.00003;
    vAlpha = clamp(alphaBase + deltaAlpha, 0.0, 1.0);
    vPetalBend = (kind > 3.5 && kind < 4.5) ? clamp(p0.w / max(p0.z, 0.01), 0.1, 1.0) : 0.0;

    rotation += rotate;
    float c = cos(rotation);
    float s = sin(rotation);
    mat2 turn = mat2(c, s, -s, c);
    vec2 quadPosition = position.xy;
    vec2 ellipseLocal = (quadPosition - 0.5) * size * 2.2;
    vec2 local = turn * ellipseLocal * max(scale, 0.01);
    start = turn * start * max(scale, 0.01);
    control = turn * control * max(scale, 0.01);
    end = turn * end * max(scale, 0.01);
    vec2 world = local + origin + vec2(dx + wobble * 0.18, dy - wobble * 0.12);
    vLocal = local;
    vEllipseLocal = ellipseLocal;
    vWidth = max(p1.z * max(scale, 0.01), 0.55);
    vStart = start;
    vControl = control;
    vEnd = end;

    vec2 clip = (world / uLogicalSize) * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vLocal;
  varying vec2 vEllipseLocal;
  varying vec2 vStart;
  varying vec2 vControl;
  varying vec2 vEnd;
  varying vec2 vSize;
  varying float vWidth;
  varying float vPetalBend;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vKind;

  float softCircle(vec2 p, float radius) {
    float d = length(p);
    return 1.0 - smoothstep(radius * 0.55, radius, d);
  }

  float segmentDistance(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
    return length(pa - ba * h);
  }

  void main() {
    float coverage = 0.0;
    if (vKind < 0.5) {
      vec2 unit = vec2(vEllipseLocal.x / max(vSize.x, 0.01), vEllipseLocal.y / max(vSize.y, 0.01));
      coverage = 1.0 - smoothstep(0.82, 1.0, length(unit));
    } else if (vKind < 1.5) {
      float d = segmentDistance(vLocal, vStart, vEnd);
      coverage = 1.0 - smoothstep(vWidth * 0.55, vWidth, d);
    } else if (vKind < 2.5) {
      float d = min(
        segmentDistance(vLocal, vStart, vControl),
        segmentDistance(vLocal, vControl, vEnd)
      );
      coverage = 1.0 - smoothstep(vWidth * 0.55, vWidth, d);
    } else if (vKind < 3.5) {
      // Soft segment stroke follows a quadratic Bezier with round caps.
      float d = 1e5;
      vec2 previous = vStart;
      for (int step = 1; step <= 8; step++) {
        float t = float(step) / 8.0;
        vec2 point = (1.0 - t) * (1.0 - t) * vStart
          + 2.0 * (1.0 - t) * t * vControl
          + t * t * vEnd;
        d = min(d, segmentDistance(vLocal, previous, point));
        previous = point;
      }
      float width = max(vWidth, 0.8);
      coverage = 1.0 - smoothstep(width * 0.35, width, d);
    } else if (vKind < 4.5) {
      // Petal silhouette: two quadratic arcs joined at the tip.
      float x = vLocal.x / max(vSize.x, 0.01);
      float y = vLocal.y / max(vSize.y, 0.01);
      float taper = sqrt(max(1.0 - x * x, 0.0));
      float petal = taper * mix(0.42, 0.95, vPetalBend) * (1.0 + vPetalBend * 0.22 * x);
      coverage = 1.0 - smoothstep(petal * 0.55, petal, abs(y));
      coverage *= 1.0 - smoothstep(0.86, 1.0, abs(x));
    } else if (vKind < 5.5) {
      coverage = softCircle(vLocal, max(vSize.x, 0.5));
    } else {
      float q = length(vLocal / max(vSize.x, 0.01));
      coverage = 1.0 - smoothstep(0.0, 1.0, q);
    }
    float alpha = coverage * vAlpha;
    if (alpha < 0.002) discard;
    gl_FragColor = vec4(vColor * alpha, alpha);
  }
`;

const BACKGROUND_VERTEX_SHADER = /* glsl */ `
  precision highp float;
  attribute vec3 position;
  varying vec2 vScreen;
  void main() {
    vScreen = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const BACKGROUND_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  uniform sampler2D uBase;
  varying vec2 vScreen;
  void main() {
    // The only fixed image used by the player. It never moves or changes.
    vec2 uv = vec2(vScreen.x, 1.0 - vScreen.y);
    gl_FragColor = vec4(texture2D(uBase, uv).rgb, 1.0);
  }
`;

function makeDataTexture(data, width, height, renderer) {
    const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
    texture.needsUpdate = true;
    renderer.initTexture(texture);
  return texture;
}

function paddedTextureDimensions(floats, maxSize) {
  const texels = Math.ceil(floats.length / 4);
  const width = Math.min(maxSize, 2048, Math.max(1, Math.ceil(Math.sqrt(texels))));
  const height = Math.ceil(texels / width);
  const padded = new Float32Array(width * height * 4);
  padded.set(floats);
  return { padded, width, height };
}

export class StrokePlayer {
  constructor(canvas, { onStatus, onStats, onError } = {}) {
    this.canvas = canvas;
    this.onStatus = onStatus || (() => {});
    this.onStats = onStats || (() => {});
    this.onError = onError || (() => {});
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: false, alpha: false, preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.debug.onShaderError = (gl, program, vertex, fragment) => {
      this.onError('Three.js shader 编译失败', {
        program: gl.getProgramInfoLog(program),
        vertex: gl.getShaderInfoLog(vertex),
        fragment: gl.getShaderInfoLog(fragment),
      });
    };
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.clock = new THREE.Clock();
    this.playing = true;
    this.elapsed = 0;
    this.loaded = false;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement || canvas);
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  async load(url) {
    this.onStatus('正在下载分层绘图指令…');
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`无法加载 ${url}: HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);
    this.lastBufferInfo = {
      bytes: buffer.byteLength,
      magic: new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(8, buffer.byteLength))),
      header: buffer.byteLength >= 20 ? {
        version: view.getUint16(8, true),
        keyframes: view.getUint16(10, true),
        jsonLength: view.getUint32(12, true),
        strokeCount: view.getUint32(16, true),
      } : null,
    };
    const meta = this.parse(buffer);
    if (meta.background?.path) await this.loadBackground(meta.background.path);
    this.overlay = await DynamicOverlay.load(
      'assets/temporal_runtime.bin?v=9',
      'assets/temporal_runtime.manifest.json?v=9',
    );
    this.overlay.bind(this.overlayCanvas);
    this.overlay.draw(0);
    this.onStats({
      ...this.meta,
      layers: this.layerCounts,
      regionCounts: this.regionCounts,
      modelCounts: this.modelCounts,
      bytes: this.lastBufferInfo.bytes,
      temporal: this.overlay.manifest,
    });
    this.loaded = true;
    this.onStatus('运行时程序化绘制中');
  }

  async loadBackground(url) {
    const texture = await new THREE.TextureLoader().loadAsync(url);
    texture.colorSpace = THREE.NoColorSpace;
    texture.flipY = false;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([
      -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0,
    ], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    const material = new THREE.RawShaderMaterial({
      vertexShader: BACKGROUND_VERTEX_SHADER,
      fragmentShader: BACKGROUND_FRAGMENT_SHADER,
      uniforms: { uBase: { value: texture } },
      depthTest: false,
      depthWrite: false,
      transparent: false,
    });
    const background = new THREE.Mesh(geometry, material);
    background.frustumCulled = false;
    background.renderOrder = -1;
    this.scene.add(background);
  }

  parse(buffer) {
    const result = parseInstructions(buffer);
    const {
      meta, params, colors, motion, keys, layerCounts,
      regionCounts, modelCounts, strokeCount, bytes,
      staticCount, dynamicCount, dynamicStart,
    } = result;
    this.meta = meta;
    const maxTextureSize = this.renderer.capabilities.maxTextureSize || 2048;
    const paramTexture = paddedTextureDimensions(params, maxTextureSize);
    const colorTexture = paddedTextureDimensions(colors, maxTextureSize);
    const motionTexture = paddedTextureDimensions(motion, maxTextureSize);
    const keyTexture = paddedTextureDimensions(keys, maxTextureSize);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([
      -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
    ], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    const staticIndices = new Float32Array(staticCount);
    const dynamicIndices = new Float32Array(dynamicCount);
    for (let i = 0; i < staticCount; i += 1) staticIndices[i] = i;
    for (let i = 0; i < dynamicCount; i += 1) dynamicIndices[i] = dynamicStart + i;
    geometry.setAttribute('aIndex', new THREE.InstancedBufferAttribute(staticIndices, 1));
    geometry.instanceCount = staticCount;
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);

    this.uniforms = {
      uParams: { value: makeDataTexture(paramTexture.padded, paramTexture.width, paramTexture.height, this.renderer) },
      uColor: { value: makeDataTexture(colorTexture.padded, colorTexture.width, colorTexture.height, this.renderer) },
      uMotion: { value: makeDataTexture(motionTexture.padded, motionTexture.width, motionTexture.height, this.renderer) },
      uKeys: { value: makeDataTexture(keyTexture.padded, keyTexture.width, keyTexture.height, this.renderer) },
      uParamSize: { value: new THREE.Vector2(paramTexture.width, paramTexture.height) },
      uColorSize: { value: new THREE.Vector2(colorTexture.width, colorTexture.height) },
      uMotionSize: { value: new THREE.Vector2(motionTexture.width, motionTexture.height) },
      uKeySize: { value: new THREE.Vector2(keyTexture.width, keyTexture.height) },
      uTime: { value: 0 },
      uFrame: { value: 0 },
      uLogicalSize: { value: new THREE.Vector2(this.meta.width, this.meta.height) },
    };
    const material = new THREE.RawShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: this.uniforms,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    const dynamicGeometry = geometry.clone();
    dynamicGeometry.setAttribute('aIndex', new THREE.InstancedBufferAttribute(dynamicIndices, 1));
    // Static WebGL primitives stay fixed in unified coordinates. Frozen
    // dynamic instances are disabled; their current-frame detail comes from
    // the VTSK0002 Canvas foreground instead.
    dynamicGeometry.instanceCount = 0;
    dynamicGeometry.boundingSphere = geometry.boundingSphere;

    // One material and shared uniforms keep the fixed WebGL reconstruction and
    // the frame-indexed Canvas foreground on the same timeline.
    this.staticMaterial = material;
    this.dynamicMaterial = material;
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    this.dynamicMesh = new THREE.Mesh(dynamicGeometry, material);
    this.dynamicMesh.frustumCulled = false;
    this.dynamicMesh.visible = false;
    this.dynamicMesh.renderOrder = 2;
    const strokeLayer = new THREE.Group();
    strokeLayer.add(this.mesh);
    strokeLayer.add(this.dynamicMesh);
    this.scene.add(strokeLayer);
    this.materials = [material];
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.className = 'dynamic-overlay';
    overlayCanvas.setAttribute('aria-hidden', 'true');
    (this.canvas.parentElement || document.body).appendChild(overlayCanvas);
    this.overlayCanvas = overlayCanvas;
    this.strokeCount = strokeCount;
    this.staticCount = staticCount;
    this.dynamicCount = dynamicCount;
    this.layerCounts = layerCounts;
    this.regionCounts = regionCounts;
    this.modelCounts = modelCounts;
    this.onStats({ ...this.meta, layers: layerCounts, regionCounts, modelCounts, bytes });
    this.resize();
    return this.meta;
  }

  resize() {
    const parent = this.canvas.parentElement;
    const width = Math.max(1, Math.floor(parent ? parent.clientWidth : this.canvas.clientWidth));
    const height = Math.max(1, Math.floor(parent ? parent.clientHeight : this.canvas.clientHeight));
    this.renderer.setSize(width, height, false);
  }

  setTime(seconds) {
    if (!this.meta) return;
    this.elapsed = Math.max(0, Math.min(this.meta.duration, seconds));
    const normalized = this.elapsed / this.meta.duration;
    this.uniforms.uTime.value = normalized;
    this.uniforms.uFrame.value = normalized * (this.meta.frame_count - 1);
    for (const material of this.materials) {
      material.uniforms.uTime.value = normalized;
      material.uniforms.uFrame.value = normalized * (this.meta.frame_count - 1);
      material.uniformsNeedUpdate = true;
    }
    if (this.overlay) this.overlay.draw(this.elapsed);
    if (this.loaded) this.renderer.render(this.scene, this.camera);
  }

  toggle() {
    this.playing = !this.playing;
    return this.playing;
  }

  drawFrame() {
    this.setTime(this.elapsed);
  }

  animate() {
    requestAnimationFrame(this.animate);
    const delta = Math.min(this.clock.getDelta(), 0.1);
    if (this.playing && this.loaded) {
      this.elapsed = (this.elapsed + delta) % this.meta.duration;
      this.drawFrame();
    }
  }
}
