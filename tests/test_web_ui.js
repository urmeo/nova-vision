const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = readFileSync(join(__dirname, '..', 'docs', 'index.html'), 'utf8');
const script = readFileSync(join(__dirname, '..', 'docs', 'app.js'), 'utf8');

class Element {
    constructor(tag = 'div', downloads = []) {
        this.tag = tag;
        this.downloads = downloads;
        this.children = [];
        this.style = {};
        this.value = '';
        this.listeners = new Map();
        this.attributes = new Map();
        this.parts = new Map();
        this.classes = new Set();
        this.classList = {
            add: (...names) => names.forEach(name => this.classes.add(name)),
            remove: (...names) => names.forEach(name => this.classes.delete(name)),
            contains: name => this.classes.has(name),
            toggle: (name, force = !this.classes.has(name)) => {
                if (force) this.classes.add(name);
                else this.classes.delete(name);
                return force;
            }
        };
    }
    set className(value) { this.classes = new Set(value.split(/\s+/)); }
    set innerHTML(value) { this.children = []; }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    querySelector(selector) {
        if (!this.parts.has(selector)) this.parts.set(selector, new Element());
        return this.parts.get(selector);
    }
    addEventListener(name, listener) {
        if (!this.listeners.has(name)) this.listeners.set(name, []);
        this.listeners.get(name).push(listener);
    }
    click() {
        if (this.tag === 'a') this.downloads.push({ image: this.href, filename: this.download });
        return Promise.all((this.listeners.get('click') || []).map(listener => listener()));
    }
    focus() {}
}

function createApp(fetch) {
    const elements = new Map();
    const downloads = [];
    const getElementById = id => {
        if (!elements.has(id)) {
            assert(html.includes(`id="${id}"`), `Missing HTML element: ${id}`);
            elements.set(id, new Element());
        }
        return elements.get(id);
    };
    getElementById('liveEmotionBars').children = Array.from({ length: 7 }, () => new Element());
    const context = vm.createContext({
        document: {
            documentElement: { dataset: {} },
            getElementById,
            createElement: tag => new Element(tag, downloads),
            querySelectorAll: () => []
        },
        window: { scrollY: 0, addEventListener() {} },
        performance: { now: () => 1000 },
        console: { error: (...messages) => assert.fail(messages.join(' ')) },
        AbortController,
        setTimeout,
        clearTimeout,
        fetch
    });
    vm.runInContext(script, context, { filename: 'docs/app.js' });
    const analyze = vm.runInContext('analyzeLive', context);
    return { analyze, getElementById, downloads };
}

test('analyzed text produces a downloadable image', async () => {
    const requests = [];
    const text = 'A peaceful sunset over a quiet lake';
    const image = 'data:image/png;base64,'
        + 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC';
    const app = createApp(async (url, options) => {
        assert.equal(options.headers['X-API-Token'], 'test-token');
        const body = JSON.parse(options.body);
        requests.push({ url, body });
        return {
            ok: true,
            json: async () => ({
                success: true,
                image,
                original_text: body.text,
                prompt: `${body.text}, landscape photography`,
                primary_emotion: 'joy',
                confidence: 70,
                emotions: [{ name: 'joy', score: 70 }, { name: 'neutral', score: 30 }],
                valence: 0.6,
                arousal: 0.4,
                style: body.style || 'nature',
                seed: body.seed ?? 17,
                backend: 'null',
                timestamp: '2026-10-22T00:00:00Z'
            })
        };
    });
    const element = app.getElementById;
    element('emotionInput').value = text;
    element('styleSelect').value = 'nature';
    element('seedInput').value = '17';
    element('tokenInput').value = 'test-token';

    await app.analyze(text);
    assert.equal(element('liveStatusText').textContent, 'Joy');
    await element('generateBtn').click();

    assert.deepEqual(requests, [
        { url: '/api/analyze', body: { text } },
        { url: '/api/generate', body: { text, style: 'nature', seed: 17 } }
    ]);
    assert.equal(element('generatedImage').src, image);
    assert.equal(element('generationBackend').textContent, 'Test image · null backend');
    assert.equal(element('historyCount').textContent, '1 of 4');
    assert.equal(element('generateBtn').disabled, false);
    await element('downloadImageBtn').click();
    assert.equal(app.downloads.length, 1);
    assert.equal(app.downloads[0].image, image);
    assert.match(app.downloads[0].filename, /^novavision_joy_.*\.png$/);
});
