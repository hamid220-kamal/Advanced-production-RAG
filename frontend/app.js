const API_BASE = 'http://127.0.0.1:8000/api/v1';

// Generate a unique session ID for conversation memory
const sessionId = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2);

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
        const response = await fetch(`${API_BASE}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                user_query: query, 
                use_hyde: useHyde, 
                top_k: 3,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to execute query');
        }

        resultsContainer.classList.remove('hidden');
        answerContent.innerHTML = "<p></p>";
        const pTag = answerContent.querySelector("p");
        let answerText = "";

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            let eventEndIndex;
            while ((eventEndIndex = buffer.indexOf('\n\n')) >= 0) {
                const eventStr = buffer.substring(0, eventEndIndex);
                buffer = buffer.substring(eventEndIndex + 2);
                
                const eventMatch = eventStr.match(/event:\s*(.*)/);
                const dataMatch = eventStr.match(/data:\s*(.*)/);
                
                if (eventMatch && dataMatch) {
                    const currentEvent = eventMatch[1];
                    const dataStr = dataMatch[1];
                    
                    if (currentEvent === 'metadata') {
                        const meta = JSON.parse(dataStr);
                        renderSources(meta);
                    } else if (currentEvent === 'token') {
                        const tokenData = JSON.parse(dataStr);
                        answerText += tokenData.token;
                        pTag.textContent = answerText;
                    }
                }
            }
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error(error);
    } finally {
        // UI Reset state
        queryBtn.disabled = false;
        queryBtnIcon.classList.remove('hidden');
        querySpinner.classList.add('hidden');
    }
});

function renderSources(results) {
    sourcesContent.innerHTML = '';
    results.forEach((result, index) => {
        const sourceHtml = `
            <div class="source-item" style="animation: slideUp ${0.3 + (index * 0.1)}s ease">
                <div class="source-meta">
                    <span class="source-name">📄 ${result.source}</span>
                    <span class="source-chunk">Chunk ID: ${result.chunk_id.substring(0, 8)}...</span>
                </div>
                <div class="source-content">
                    "${result.content}"
                </div>
            </div>
        `;
        sourcesContent.insertAdjacentHTML('beforeend', sourceHtml);
    });
}
