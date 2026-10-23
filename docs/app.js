const isPreview = document.documentElement.dataset.preview === 'true';
const navbar = document.getElementById('navbar');
const navLinks = document.querySelectorAll('.nav-links a');
const sections = document.querySelectorAll('section[id], .analysis-section[id], .prompt-engineering-section[id]');

function updateActiveNav() {
    const scrollY = window.scrollY + 100;
    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === '#' + currentSection);
    });
}

window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 50);
    updateActiveNav();
}, { passive: true });

updateActiveNav();

const liveStatus = document.getElementById('liveStatus');
const liveStatusText = document.getElementById('liveStatusText');
const liveEmotionBars = document.getElementById('liveEmotionBars');
const liveTip = document.getElementById('liveTip');
let debounceTimer = null;
let liveAnalysisController = null;

const emotionOrder = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral'];
const minTextLength = 3;
const maxTextLength = 2000;

function formatEmotionName(emotion) {
    return emotion.charAt(0).toUpperCase() + emotion.slice(1);
}

function cancelLiveAnalysis() {
    if (liveAnalysisController) liveAnalysisController.abort();
    liveAnalysisController = null;
}

function resetToEmpty() {
    cancelLiveAnalysis();
    liveEmotionBars.classList.add('muted');
    liveTip.classList.remove('visible');
    liveStatus.classList.remove('analyzing', 'active');
    liveStatusText.textContent = isPreview ? 'Requires the Python server' : 'Ready';

    for (const bar of liveEmotionBars.children) {
        bar.querySelector('.live-emotion-name').classList.remove('dominant');
        const fill = bar.querySelector('.live-emotion-fill');
        fill.style.width = '2px';
        fill.className = 'live-emotion-fill muted';
        bar.querySelector('.live-emotion-value').textContent = '—';
    }
}

function updateLiveEmotions(emotions, primaryEmotion, confidence) {
    liveEmotionBars.classList.remove('muted');
    liveTip.classList.add('visible');
    liveStatus.classList.remove('analyzing');
    liveStatus.classList.add('active');

    const emotionMap = Object.fromEntries(emotions.map(emotion => [emotion.name, emotion.score]));
    const showDominant = confidence > 30;

    emotionOrder.forEach((emotion, index) => {
        const bar = liveEmotionBars.children[index];
        if (bar) {
            const nameEl = bar.querySelector('.live-emotion-name');
            const fill = bar.querySelector('.live-emotion-fill');
            const value = bar.querySelector('.live-emotion-value');
            const score = emotionMap[emotion] ?? 0;

            fill.style.width = score > 0 ? Math.max(2, score) + '%' : '2px';

            const isPrimary = emotion === primaryEmotion && showDominant;
            fill.className = 'live-emotion-fill' + (isPrimary ? ' primary' : '');
            nameEl.classList.toggle('dominant', isPrimary);
            value.textContent = score.toFixed(1) + '%';
        }
    });

    liveStatusText.textContent = showDominant ? formatEmotionName(primaryEmotion) : 'Scores ready';
}

function apiHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    const apiToken = tokenInput.value.trim();
    if (apiToken) headers['X-API-Token'] = apiToken;
    return headers;
}

async function analyzeLive(text) {
    if (isPreview) return;
    text = text.trim();
    const textLength = [...text].length;
    if (textLength < minTextLength || textLength > maxTextLength) {
        resetToEmpty();
        liveStatusText.textContent = `Use ${minTextLength}–${maxTextLength} characters`;
        return;
    }
    resetToEmpty();
    liveStatus.classList.add('analyzing');
    liveStatus.classList.remove('active');
    liveStatusText.textContent = 'Analyzing...';

    const controller = new AbortController();
    liveAnalysisController = controller;
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify({ text }),
            signal: controller.signal
        });

        const data = await response.json();
        if (controller.signal.aborted) return;

        if (response.ok && hasValidAnalysis(data)) {
            updateLiveEmotions(data.emotions, data.primary_emotion, data.confidence);
        } else {
            resetToEmpty();
            liveStatusText.textContent = 'Analysis unavailable';
        }
    } catch (error) {
        if (controller.signal.aborted) return;
        console.error('Live analysis error:', error);
        resetToEmpty();
        liveStatusText.textContent = 'Analysis unavailable';
    } finally {
        if (liveAnalysisController === controller) liveAnalysisController = null;
    }
}

const emotionInput = document.getElementById('emotionInput');

resetToEmpty();

emotionInput.addEventListener('input', () => {
    if (isPreview) return;
    clearTimeout(debounceTimer);

    const text = emotionInput.value.trim();
    const textLength = [...text].length;
    const valid = textLength >= minTextLength && textLength <= maxTextLength;
    resetToEmpty();
    generateBtn.classList.toggle('ready', valid);
    if (!valid) {
        if (textLength > maxTextLength) {
            liveStatusText.textContent = `Use ${minTextLength}–${maxTextLength} characters`;
        }
        return;
    }
    liveStatusText.textContent = 'Waiting for analysis';
    debounceTimer = setTimeout(() => analyzeLive(text), 500);
});

let currentGenerationData = null;

function downloadPng(image, filename) {
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.download = `${filename}_${timestamp}.png`;
    link.href = image;
    link.click();
}

function downloadImage() {
    if (currentGenerationData) {
        downloadPng(currentGenerationData.image, `novavision_${currentGenerationData.primary_emotion}`);
    }
}

function wrapReportText(ctx, text, maxWidth) {
    const lines = [];
    for (const paragraph of text.split(/\r\n|[\r\n\u2028\u2029]/)) {
        let line = '';
        for (const word of paragraph.split(/\s+/).filter(Boolean)) {
            const candidate = line ? line + ' ' + word : word;
            if (ctx.measureText(candidate).width <= maxWidth) {
                line = candidate;
                continue;
            }
            if (line) {
                lines.push(line);
                line = '';
            }
            for (const character of word) {
                if (line && ctx.measureText(line + character).width > maxWidth) {
                    lines.push(line);
                    line = character;
                } else {
                    line += character;
                }
            }
        }
        lines.push(line);
    }
    return lines;
}

function downloadReport() {
    if (!currentGenerationData) return;
    const generationData = currentGenerationData;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 1200;
    const maxWidth = canvas.width - 80;
    ctx.font = '14px Inter, sans-serif';
    const promptLines = wrapReportText(ctx, generationData.prompt, maxWidth);
    ctx.font = 'italic 16px Inter, sans-serif';
    const inputLines = wrapReportText(ctx, '"' + generationData.original_text + '"', maxWidth);
    const promptStartY = 920;
    const promptEndY = promptStartY + (promptLines.length - 1) * 24;
    const inputHeadingY = promptEndY + 60;
    const inputStartY = inputHeadingY + 35;
    const inputEndY = inputStartY + (inputLines.length - 1) * 26;
    canvas.height = Math.max(1600, inputEndY + 100);

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, '#6366f1');
    gradient.addColorStop(1, '#8b5cf6');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, 80);

    function drawText(text, x, y, font = '14px Inter, sans-serif', color = '#64748b') {
        ctx.font = font;
        ctx.fillStyle = color;
        ctx.fillText(text, x, y);
    }

    drawText('NovaVision', 40, 52, 'bold 32px Inter, sans-serif', '#ffffff');
    ctx.textAlign = 'right';
    drawText(new Date(generationData.timestamp).toLocaleString(), canvas.width - 40, 52,
        '16px Inter, sans-serif', 'rgba(255,255,255,0.8)');
    ctx.textAlign = 'left';

    const img = new Image();
    img.onload = () => {
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0,0,0,0.1)';
        ctx.shadowBlur = 20;
        ctx.fillRect(40, 120, 700, 700);
        ctx.shadowBlur = 0;

        ctx.drawImage(img, 50, 130, 680, 680);
        drawText(backendLabel(generationData.backend), 50, 840);
        drawText('Text analysis', 780, 160, 'bold 24px Inter, sans-serif', '#0f172a');
        drawText('LEADING LABEL', 780, 200);
        drawText(formatEmotionName(generationData.primary_emotion), 780, 240,
            'bold 36px Inter, sans-serif', '#6366f1');
        drawText(generationData.confidence + '% classifier score', 780, 270,
            '16px Inter, sans-serif', '#94a3b8');

        generationData.emotions.forEach((emotion, index) => {
            const y = 320 + index * 35;
            drawText(formatEmotionName(emotion.name), 780, y, '13px Inter, sans-serif');
            ctx.fillStyle = '#e2e8f0';
            ctx.fillRect(870, y - 10, 250, 8);
            ctx.fillStyle = index === 0 ? '#6366f1' : '#cbd5e1';
            ctx.fillRect(870, y - 10, (emotion.score / 100) * 250, 8);
            drawText(emotion.score.toFixed(1) + '%', 1130, y, '13px Inter, sans-serif', '#94a3b8');
        });

        drawText('Generation prompt', 40, 880, 'bold 20px Inter, sans-serif', '#0f172a');
        promptLines.forEach((line, index) => drawText(line, 40, promptStartY + index * 24));
        drawText('Your text', 40, inputHeadingY, 'bold 20px Inter, sans-serif', '#0f172a');
        inputLines.forEach((line, index) => {
            drawText(line, 40, inputStartY + index * 26, 'italic 16px Inter, sans-serif');
        });

        ctx.fillStyle = '#e2e8f0';
        ctx.fillRect(0, canvas.height - 60, canvas.width, 60);
        ctx.textAlign = 'center';
        drawText('NovaVision · Image generation guided by emotion labels', canvas.width / 2,
            canvas.height - 25, '14px Inter, sans-serif', '#94a3b8');

        downloadPng(canvas.toDataURL('image/png'), 'novavision_report');
    };
    img.src = generationData.image;
}

document.getElementById('downloadImageBtn').addEventListener('click', downloadImage);
document.getElementById('downloadReportBtn').addEventListener('click', downloadReport);

document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        if (isPreview) return;
        emotionInput.value = chip.dataset.prompt;
        emotionInput.focus();

        const event = new Event('input', { bubbles: true });
        emotionInput.dispatchEvent(event);
    });
});

const advancedToggle = document.getElementById('advancedToggle');
const advancedContent = document.getElementById('advancedContent');
const seedInput = document.getElementById('seedInput');
const tokenInput = document.getElementById('tokenInput');
const copySeedBtn = document.getElementById('copySeedBtn');
const usedSeedDisplay = document.getElementById('usedSeedDisplay');
const lastSeedValue = document.getElementById('lastSeedValue');

advancedToggle.addEventListener('click', () => {
    const expanded = advancedContent.classList.toggle('open');
    advancedToggle.classList.toggle('open', expanded);
    advancedToggle.setAttribute('aria-expanded', String(expanded));
});

copySeedBtn.addEventListener('click', async () => {
    if (!currentGenerationData) return;
    try {
        await navigator.clipboard.writeText(String(currentGenerationData.seed));
        copySeedBtn.textContent = 'Copied';
        copySeedBtn.classList.add('copied');
    } catch {
        copySeedBtn.textContent = 'Copy unavailable';
    } finally {
        setTimeout(() => {
            copySeedBtn.textContent = 'Copy seed';
            copySeedBtn.classList.remove('copied');
        }, 2000);
    }
});

const imageHistory = [];
const maxHistoryItems = 4;
const historyThumbnails = document.getElementById('historyThumbnails');
const historyCount = document.getElementById('historyCount');

function renderHistory() {
    historyThumbnails.innerHTML = '';
    historyCount.textContent = `${imageHistory.length} of ${maxHistoryItems}`;

    imageHistory.forEach(item => {
        const thumbnail = document.createElement('button');
        thumbnail.type = 'button';
        thumbnail.className = 'history-thumbnail' + (item === currentGenerationData ? ' active' : '');
        thumbnail.title = item.original_text;
        thumbnail.setAttribute('aria-label', `Restore image: ${item.original_text}`);
        thumbnail.setAttribute('aria-pressed', String(item === currentGenerationData));
        const image = document.createElement('img');
        image.src = item.image;
        image.alt = '';
        thumbnail.appendChild(image);
        thumbnail.addEventListener('click', () => {
            displayGeneration(item);
            document.querySelectorAll('.history-thumbnail').forEach(previous => {
                const selected = previous === thumbnail;
                previous.classList.toggle('active', selected);
                previous.setAttribute('aria-pressed', String(selected));
            });
        });
        historyThumbnails.appendChild(thumbnail);
    });

    const remainingSlots = maxHistoryItems - imageHistory.length;
    for (let i = 0; i < remainingSlots; i++) {
        const placeholder = document.createElement('div');
        placeholder.className = 'history-placeholder';
        placeholder.textContent = '+';
        historyThumbnails.appendChild(placeholder);
    }
}

const generateBtn = document.getElementById('generateBtn');
const styleSelect = document.getElementById('styleSelect');
const btnText = generateBtn.querySelector('.btn-text');
const imageDisplay = document.getElementById('imageDisplay');
const generatedImage = document.getElementById('generatedImage');
const generationStatus = document.getElementById('generationStatus');
const placeholder = document.getElementById('placeholder');
const downloadActions = document.getElementById('downloadActions');

const skeletonLoader = document.getElementById('skeletonLoader');
const loadingOverlay = document.getElementById('loadingOverlay');

const generationTimeEl = document.getElementById('generationTime');
const generationTimeValue = document.getElementById('generationTimeValue');

const errorDisplay = document.getElementById('errorDisplay');
const errorText = document.getElementById('errorText');
const errorDetails = document.getElementById('errorDetails');
const retryBtn = document.getElementById('retryBtn');

const styleDescriptions = {
    'photorealistic': 'hyperrealistic photograph, 85mm lens, natural light, ultra sharp, 8k',
    'artistic': 'digital painting, expressive brushwork, rich color, dramatic lighting',
    'abstract': 'abstract art, geometric forms, bold composition, modern gallery piece',
    'nature': 'landscape photography, golden hour, atmospheric depth, fine detail',
    'dreamscape': 'surreal dreamscape, ethereal glow, volumetric light, imaginative'
};

function setGenerating(active) {
    generateBtn.disabled = active;
    generateBtn.classList.toggle('loading', active);
    btnText.textContent = active ? 'Generating...' : 'Generate image';
    imageDisplay.setAttribute('aria-busy', String(active));
    skeletonLoader.classList.toggle('active', active);
    loadingOverlay.classList.toggle('active', active);
    imageDisplay.classList.toggle('generating', active);
    placeholder.style.display = active || currentGenerationData ? 'none' : 'block';
    if (active) {
        generationStatus.textContent = 'Generating image.';
        generateBtn.classList.remove('ready');
        errorDisplay.classList.remove('visible');
    }
}

function showError(message, details = '') {
    generationStatus.textContent = '';
    errorText.textContent = message;
    errorDetails.textContent = details;
    errorDisplay.classList.add('visible');
}

const backendLabels = new Map([
    ['null', 'Test image · null backend'],
    ['diffusers', 'Local generation · diffusers backend'],
    ['hf-api', 'Hosted generation · hf-api backend']
]);

function backendLabel(backend) {
    return backendLabels.get(backend) || `Image backend · ${backend || 'unknown'}`;
}

function displayAffectValue(element, label, value, signed = false) {
    const number = Number.isFinite(value) ? value.toFixed(1) : 'n/a';
    const sign = signed && Number.isFinite(value) && value >= 0 ? '+' : '';
    element.className = `dimension-value ${label.toLowerCase()}`;
    element.innerHTML = `${label}<span class="dimension-number">(${sign}${number})</span>`;
}

function displayGeneration(data) {
    currentGenerationData = data;
    document.getElementById('generationBackend').textContent =
        backendLabel(data.backend);

    lastSeedValue.textContent = data.seed;
    usedSeedDisplay.classList.add('visible');

    generatedImage.src = data.image;
    const imageType = data.backend === 'null' ? 'Test image' : 'Generated image';
    generatedImage.alt = `${imageType} for: ${data.original_text}`;
    generatedImage.classList.add('visible');
    generationStatus.textContent = 'Image ready.';
    imageDisplay.classList.add('has-image');
    placeholder.style.display = 'none';

    generationTimeValue.textContent = data.generationSeconds;
    generationTimeEl.classList.add('visible');

    downloadActions.classList.add('visible');

    const primaryEmotionEl = document.getElementById('primaryEmotion');
    primaryEmotionEl.classList.remove('default');
    primaryEmotionEl.textContent = formatEmotionName(data.primary_emotion);
    document.getElementById('primaryConfidence').textContent =
        data.confidence + '% classifier score';

    const valence = data.valence;
    const arousal = data.arousal;
    const valenceDisplay = document.getElementById('valenceDisplay');
    const valenceLabel = valence === 0 ? 'Neutral' : valence > 0 ? 'Positive' : 'Negative';
    displayAffectValue(valenceDisplay, valenceLabel, valence, true);

    const arousalDisplay = document.getElementById('arousalDisplay');
    const arousalLabel = arousal >= 0.5 ? 'Energized' : 'Calm';
    displayAffectValue(arousalDisplay, arousalLabel, arousal);

    const emotionBars = document.getElementById('emotionBars');
    emotionBars.innerHTML = '';
    data.emotions.forEach((emotion, index) => {
        const bar = document.createElement('div');
        bar.className = 'emotion-bar';
        const name = document.createElement('span');
        name.className = 'emotion-name';
        name.textContent = formatEmotionName(emotion.name);
        const track = document.createElement('div');
        track.className = 'emotion-track';
        const fill = document.createElement('div');
        fill.className = 'emotion-fill' + (index > 0 ? ' secondary' : '');
        fill.style.width = emotion.score + '%';
        track.appendChild(fill);
        const value = document.createElement('span');
        value.className = 'emotion-value';
        value.textContent = emotion.score.toFixed(1) + '%';
        bar.append(name, track, value);
        emotionBars.appendChild(bar);
    });

    document.getElementById('userInputDisplay').textContent = data.original_text;
    document.getElementById('enhancedPromptDisplay').textContent = data.prompt;
    document.getElementById('modUserText').textContent = '"' + data.original_text + '"';
    document.getElementById('modStyleText').textContent = '"' + styleDescriptions[data.style] + '"';
}

function hasValidAnalysis(data) {
    const percentage = value => Number.isFinite(value) && value >= 0 && value <= 100;
    return data !== null && typeof data === 'object' && data.success === true
        && emotionOrder.includes(data.primary_emotion)
        && percentage(data.confidence)
        && Number.isFinite(data.valence) && data.valence >= -1 && data.valence <= 1
        && Number.isFinite(data.arousal) && data.arousal >= 0 && data.arousal <= 1
        && Array.isArray(data.emotions) && data.emotions.length > 0
        && data.emotions.every(emotion => emotion !== null && typeof emotion === 'object'
            && emotionOrder.includes(emotion.name) && percentage(emotion.score))
        && new Set(data.emotions.map(emotion => emotion.name)).size === data.emotions.length
        && data.emotions.some(emotion => emotion.name === data.primary_emotion);
}

function hasValidGeneration(data, seed) {
    const nonempty = value => typeof value === 'string' && value.trim().length > 0;
    const imagePrefix = 'data:image/png;base64,';
    return hasValidAnalysis(data)
        && nonempty(data.image) && data.image.startsWith(imagePrefix)
        && data.image.slice(imagePrefix.length).trim().length > 0
        && nonempty(data.original_text) && nonempty(data.prompt)
        && nonempty(data.backend)
        && typeof data.style === 'string' && Object.hasOwn(styleDescriptions, data.style)
        && nonempty(data.timestamp) && Number.isFinite(Date.parse(data.timestamp))
        && Number.isSafeInteger(data.seed ?? seed);
}

async function generateImage() {
    if (isPreview || generateBtn.disabled) return;
    const text = emotionInput.value.trim();
    if (!text) {
        emotionInput.focus();
        return;
    }
    const textLength = [...text].length;
    if (textLength < minTextLength || textLength > maxTextLength) {
        showError(`Enter ${minTextLength}–${maxTextLength} characters.`,
            'Describe a feeling, memory, or scene.');
        emotionInput.focus();
        return;
    }
    const userSeed = seedInput.value.trim();
    if (userSeed && !/^\d+$/.test(userSeed)) {
        showError('Seed must be a whole number.', 'Fix the seed field and try again.');
        return;
    }
    const seed = userSeed ? Number(userSeed) : Math.floor(Math.random() * 2147483647);
    if (!Number.isSafeInteger(seed)) {
        showError('Seed is too large to preserve exactly.',
            `Choose a whole number from 0 to ${Number.MAX_SAFE_INTEGER}.`);
        return;
    }

    setGenerating(true);

    const startTime = performance.now();

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify({
                text,
                style: styleSelect.value,
                seed
            })
        });

        const data = await response.json();

        if (!response.ok || !hasValidGeneration(data, seed)) {
            const details = typeof data?.error === 'string' ? data.error
                : response.ok ? 'The server returned an incomplete generation response.'
                    : `The server returned HTTP ${response.status}.`;
            showError('Generation failed. Try again.', details);
            return;
        }

        const endTime = performance.now();
        const generationSeconds = ((endTime - startTime) / 1000).toFixed(1);

        const generation = { ...data, seed: data.seed ?? seed, generationSeconds };
        displayGeneration(generation);
        imageHistory.unshift(generation);
        imageHistory.splice(maxHistoryItems);
        renderHistory();

    } catch (error) {
        console.error('Generation request failed:', error);
        showError('Could not complete the request.', 'Check the server connection and try again.');
    } finally {
        setGenerating(false);
    }
}

generateBtn.addEventListener('click', generateImage);

document.getElementById('regenerateBtn').addEventListener('click', () => {
    if (isPreview || !currentGenerationData || generateBtn.disabled) return;
    clearTimeout(debounceTimer);
    cancelLiveAnalysis();
    emotionInput.value = currentGenerationData.original_text;
    styleSelect.value = currentGenerationData.style;
    updateLiveEmotions(currentGenerationData.emotions, currentGenerationData.primary_emotion,
        currentGenerationData.confidence);
    let seed = Math.floor(Math.random() * 2147483647);
    if (seed === currentGenerationData.seed) seed = (seed + 1) % 2147483647;
    seedInput.value = String(seed);
    return generateImage();
});

retryBtn.addEventListener('click', generateImage);

if (isPreview) {
    document.getElementById('previewNotice').hidden = false;
    document.querySelectorAll('#try-it button, #try-it input, #try-it textarea, #try-it select')
        .forEach(control => { control.disabled = true; });
    btnText.textContent = 'Run locally to generate images';
    emotionInput.placeholder = 'Text analysis and image generation require the Python server.';
    placeholder.querySelector('.placeholder-text').textContent = 'Image generation is unavailable in this preview.';
    document.getElementById('generationBackend').textContent = 'Static preview';
}
