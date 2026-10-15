const isPreview = document.documentElement.dataset.preview === 'true';
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');
const navbar = document.getElementById('navbar');
const heroTitle = document.querySelector('.app-header-title');
const heroSubtitle = document.querySelector('.app-header-subtitle');
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
    const scrollY = window.scrollY;

    navbar.classList.toggle('scrolled', scrollY > 50);

    if (!reducedMotion?.matches && scrollY < 400) {
        const parallaxAmount = scrollY * 0.15;
        const opacityAmount = Math.max(0, 1 - (scrollY / 300));
        if (heroTitle) {
            heroTitle.style.transform = `translateY(${parallaxAmount}px)`;
            heroTitle.style.opacity = opacityAmount;
        }
        if (heroSubtitle) {
            heroSubtitle.style.transform = `translateY(${parallaxAmount * 0.8}px)`;
            heroSubtitle.style.opacity = opacityAmount;
        }
    }

    updateActiveNav();
}, { passive: true });

updateActiveNav();

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

const liveStatus = document.getElementById('liveStatus');
const liveStatusText = document.getElementById('liveStatusText');
const liveEmotionBars = document.getElementById('liveEmotionBars');
const liveTip = document.getElementById('liveTip');
let debounceTimer = null;
let liveAnalysisController = null;
let liveAnalysisVersion = 0;

const emotionOrder = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral'];
const minTextLength = 3;
const maxTextLength = 2000;

function formatEmotionName(emotion) {
    return emotion.charAt(0).toUpperCase() + emotion.slice(1);
}

function cancelLiveAnalysis() {
    liveAnalysisVersion += 1;
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
        bar.querySelector('.live-emotion-value').textContent = isPreview ? '—' : '0%';
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
            const score = Math.max(0, Math.min(100, Number(emotionMap[emotion]) || 0));

            fill.style.width = score > 0 ? Math.max(2, score) + '%' : '2px';

            const isPrimary = emotion === primaryEmotion && showDominant;
            fill.className = 'live-emotion-fill' + (isPrimary ? ' primary' : '');
            nameEl.classList.toggle('dominant', isPrimary);
            value.textContent = score.toFixed(1) + '%';
        }
    });

    liveStatusText.textContent = showDominant ? formatEmotionName(primaryEmotion) : 'Scores ready';
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
    cancelLiveAnalysis();
    liveStatus.classList.add('analyzing');
    liveStatus.classList.remove('active');
    liveStatusText.textContent = 'Analyzing...';

    const controller = new AbortController();
    const version = liveAnalysisVersion;
    liveAnalysisController = controller;
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
            signal: controller.signal
        });

        const data = await response.json();
        if (controller.signal.aborted || version !== liveAnalysisVersion) return;

        if (data.success) {
            updateLiveEmotions(data.emotions, data.primary_emotion, data.confidence);
        } else {
            liveStatus.classList.remove('analyzing');
            liveStatusText.textContent = 'Analysis unavailable';
        }
    } catch (error) {
        if (controller.signal.aborted || version !== liveAnalysisVersion) return;
        console.error('Live analysis error:', error);
        liveStatus.classList.remove('analyzing');
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

    if (textLength < minTextLength) {
        resetToEmpty();
        generateBtn.classList.remove('ready');
        return;
    }
    if (textLength > maxTextLength) {
        resetToEmpty();
        generateBtn.classList.remove('ready');
        liveStatusText.textContent = `Use ${minTextLength}–${maxTextLength} characters`;
        return;
    }

    cancelLiveAnalysis();
    generateBtn.classList.add('ready');
    debounceTimer = setTimeout(() => analyzeLive(text), 500);
});

let currentGenerationData = null;

function downloadImage() {
    if (!currentGenerationData) return;

    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.download = `novavision_${currentGenerationData.primary_emotion}_${timestamp}.png`;
    link.href = currentGenerationData.image;
    link.click();
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

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 32px Inter, sans-serif';
    ctx.fillText('NovaVision', 40, 52);

    ctx.font = '16px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    const date = new Date(generationData.timestamp).toLocaleString();
    ctx.textAlign = 'right';
    ctx.fillText(date, canvas.width - 40, 52);
    ctx.textAlign = 'left';

    const img = new Image();
    img.onload = () => {

        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0,0,0,0.1)';
        ctx.shadowBlur = 20;
        ctx.fillRect(40, 120, 700, 700);
        ctx.shadowBlur = 0;

        ctx.drawImage(img, 50, 130, 680, 680);
        ctx.font = '14px Inter, sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.fillText(backendLabel(generationData.backend), 50, 840);

        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 24px Inter, sans-serif';
        ctx.fillText('Text analysis', 780, 160);

        ctx.font = '14px Inter, sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.fillText('LEADING LABEL', 780, 200);

        ctx.font = 'bold 36px Inter, sans-serif';
        ctx.fillStyle = '#6366f1';
        ctx.fillText(formatEmotionName(generationData.primary_emotion), 780, 240);

        ctx.font = '16px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(generationData.confidence + '% classifier score', 780, 270);

        let yPos = 320;
        generationData.emotions.forEach((emotion, index) => {
            ctx.font = '13px Inter, sans-serif';
            ctx.fillStyle = '#64748b';
            ctx.fillText(formatEmotionName(emotion.name), 780, yPos);

            ctx.fillStyle = '#e2e8f0';
            ctx.fillRect(870, yPos - 10, 250, 8);

            ctx.fillStyle = index === 0 ? '#6366f1' : '#cbd5e1';
            ctx.fillRect(870, yPos - 10, (emotion.score / 100) * 250, 8);

            ctx.fillStyle = '#94a3b8';
            ctx.fillText((Number(emotion.score) || 0).toFixed(1) + '%', 1130, yPos);

            yPos += 35;
        });

        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 20px Inter, sans-serif';
        ctx.fillText('Generation prompt', 40, 880);

        ctx.font = '14px Inter, sans-serif';
        ctx.fillStyle = '#64748b';

        promptLines.forEach((line, index) => {
            ctx.fillText(line, 40, promptStartY + index * 24);
        });

        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 20px Inter, sans-serif';
        ctx.fillText('Your text', 40, inputHeadingY);

        ctx.font = 'italic 16px Inter, sans-serif';
        ctx.fillStyle = '#64748b';
        inputLines.forEach((line, index) => {
            ctx.fillText(line, 40, inputStartY + index * 26);
        });

        ctx.fillStyle = '#e2e8f0';
        ctx.fillRect(0, canvas.height - 60, canvas.width, 60);

        ctx.font = '14px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'center';
        ctx.fillText('NovaVision · Image generation guided by emotion labels', canvas.width / 2, canvas.height - 25);

        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        link.download = `novavision_report_${timestamp}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
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

function addToHistory(data) {
    imageHistory.unshift(data);

    if (imageHistory.length > maxHistoryItems) {
        imageHistory.pop();
    }

    renderHistory();
}

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

function startLoadingAnimation() {
    errorDisplay.classList.remove('visible');
    skeletonLoader.classList.add('active');
    loadingOverlay.classList.add('active');
    imageDisplay.classList.add('generating');

    placeholder.style.display = 'none';
}

function stopLoadingAnimation() {
    skeletonLoader.classList.remove('active');
    loadingOverlay.classList.remove('active');
    imageDisplay.classList.remove('generating');
}

function showError(message, details = '') {
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
    generatedImage.classList.add('visible');
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

    const valence = Number(data.valence);
    const arousal = Number(data.arousal);
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
        fill.style.width = Math.max(0, Math.min(100, Number(emotion.score) || 0)) + '%';
        track.appendChild(fill);
        const value = document.createElement('span');
        value.className = 'emotion-value';
        value.textContent = (Number(emotion.score) || 0).toFixed(1) + '%';
        bar.append(name, track, value);
        emotionBars.appendChild(bar);
    });

    const promptTextEl = document.getElementById('promptText');
    promptTextEl.classList.remove('default');
    promptTextEl.textContent = '"' + data.prompt + '"';

    document.getElementById('userInputDisplay').textContent = data.original_text;
    document.getElementById('enhancedPromptDisplay').textContent = data.prompt;
    document.getElementById('modUserText').textContent = '"' + data.original_text + '"';
    document.getElementById('modStyleText').textContent = '"' + styleDescriptions[data.style] + '"';
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

    generateBtn.disabled = true;
    generateBtn.classList.remove('ready');
    generateBtn.classList.add('loading');
    btnText.textContent = 'Generating...';

    startLoadingAnimation();

    const startTime = performance.now();

    try {
        const generateHeaders = { 'Content-Type': 'application/json' };
        const apiToken = tokenInput.value.trim();
        if (apiToken) generateHeaders['X-API-Token'] = apiToken;
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: generateHeaders,
            body: JSON.stringify({
                text,
                style: styleSelect.value,
                seed
            })
        });

        const data = await response.json();

        if (data.error) {
            showError('Generation failed. Try again.', data.error);
            return;
        }

        const endTime = performance.now();
        const generationSeconds = ((endTime - startTime) / 1000).toFixed(1);

        const generation = { ...data, seed: data.seed ?? seed, generationSeconds };
        displayGeneration(generation);
        addToHistory(generation);

    } catch (error) {
        console.error('Generation request failed:', error);
        showError('Could not complete the request.', 'Check the server connection and try again.');
    } finally {
        stopLoadingAnimation();
        placeholder.style.display = currentGenerationData ? 'none' : 'block';
        generateBtn.disabled = false;
        generateBtn.classList.remove('loading');
        btnText.textContent = 'Generate image';
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
    document.getElementById('saved-results').hidden = false;
    document.querySelectorAll('[data-preview-src]').forEach(image => {
        image.src = image.dataset.previewSrc;
    });
    document.querySelectorAll('#try-it button, #try-it input, #try-it textarea, #try-it select')
        .forEach(control => { control.disabled = true; });
    btnText.textContent = 'Run locally to generate images';
    emotionInput.placeholder = 'Text analysis and image generation require the Python server.';
    placeholder.querySelector('.placeholder-text').textContent = 'Image generation is unavailable in this preview.';
    document.getElementById('generationBackend').textContent = 'Static preview';
    document.getElementById('primaryEmotion').textContent = 'Not run';
    document.getElementById('valenceDisplay').textContent = '—';
    document.getElementById('arousalDisplay').textContent = '—';
    document.querySelectorAll('#emotionBars .emotion-value').forEach(value => {
        value.textContent = '—';
    });
}
