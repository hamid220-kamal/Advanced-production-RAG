const API_BASE = 'http://127.0.0.1:8000/api/v1';

// DOM Elements
const ingestForm = document.getElementById('ingest-form');
const ingestBtn = document.getElementById('ingest-btn');
const ingestBtnText = ingestBtn.querySelector('.btn-text');
const ingestSpinner = ingestBtn.querySelector('.spinner');
const ingestStatus = document.getElementById('ingest-status');

const queryForm = document.getElementById('query-form');
const queryBtn = document.getElementById('query-btn');
const queryBtnIcon = queryBtn.querySelector('svg');
const querySpinner = queryBtn.querySelector('.spinner');
const resultsContainer = document.getElementById('results-container');
const answerContent = document.getElementById('answer-content');
const sourcesContent = document.getElementById('sources-content');

// Helper to show status message
function showStatus(element, message, type) {
    element.textContent = message;
    element.className = `status-msg status-${type}`;
    element.classList.remove('hidden');
    setTimeout(() => {
        element.classList.add('hidden');
    }, 5000);
}

// Ingest Form Submit Handler
ingestForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const source = document.getElementById('doc-source').value;
    const text = document.getElementById('doc-text').value;

    if (!text.trim()) {
        showStatus(ingestStatus, 'Please enter some document text.', 'error');
        return;
    }

    // UI Loading state
    ingestBtn.disabled = true;
    ingestBtnText.textContent = 'Ingesting...';
    ingestSpinner.classList.remove('hidden');

    try {
        const response = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source })
        });

        const data = await response.json();

        if (response.ok) {
            showStatus(ingestStatus, `Success! Indexed ${data.chunks_indexed} chunks.`, 'success');
            document.getElementById('doc-text').value = ''; // clear text area
        } else {
            showStatus(ingestStatus, `Error: ${data.detail || 'Failed to ingest'}`, 'error');
        }
    } catch (error) {
        showStatus(ingestStatus, `Network error: Make sure the server is running.`, 'error');
        console.error(error);
    } finally {
        // UI Reset state
        ingestBtn.disabled = false;
        ingestBtnText.textContent = 'Ingest Document';
        ingestSpinner.classList.add('hidden');
    }
});

// Query Form Submit Handler
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('user-query').value;
    const useHyde = document.getElementById('use-hyde').checked;

    if (!query.trim()) return;

    // UI Loading state
    queryBtn.disabled = true;
    queryBtnIcon.classList.add('hidden');
    querySpinner.classList.remove('hidden');
    resultsContainer.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_query: query, use_hyde: useHyde, top_k: 3 })
        });

        const data = await response.json();

        if (response.ok) {
            renderResults(data);
        } else {
            alert(`Error: ${data.detail || 'Failed to execute query'}`);
        }
    } catch (error) {
        alert('Network error: Make sure the server is running.');
        console.error(error);
    } finally {
        // UI Reset state
        queryBtn.disabled = false;
        queryBtnIcon.classList.remove('hidden');
        querySpinner.classList.add('hidden');
    }
});

function renderResults(data) {
    // Render Answer
    answerContent.innerHTML = `<p>${data.generated_answer}</p>`;

    // Render Sources
    sourcesContent.innerHTML = '';
    data.results.forEach((result, index) => {
        const sourceHtml = `
            <div class="source-item" style="animation: slideUp ${0.3 + (index * 0.1)}s ease">
                <div class="source-meta">
                    <span class="source-name">📄 ${result.metadata.source}</span>
                    <span class="source-chunk">Chunk ID: ${result.metadata.chunk_id.substring(0, 8)}...</span>
                </div>
                <div class="source-content">
                    "${result.content}"
                </div>
            </div>
        `;
        sourcesContent.insertAdjacentHTML('beforeend', sourceHtml);
    });

    resultsContainer.classList.remove('hidden');
}
