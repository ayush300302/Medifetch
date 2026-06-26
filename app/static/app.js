document.addEventListener("DOMContentLoaded", () => {
    // ── UI Elements ──
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");
    const guidelinesList = document.getElementById("guidelines-list");
    const docCountBadge = document.getElementById("doc-count");
    const clearDbBtn = document.getElementById("clear-db-btn");
    
    const chatFrame = document.getElementById("chat-frame");
    const welcomeContainer = document.getElementById("welcome-container");
    const messagesList = document.getElementById("messages-list");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    
    const suggestBtns = document.querySelectorAll(".suggest-btn");
    
    // Citation Inspector elements
    const citationDrawer = document.getElementById("citation-drawer");
    const drawerOverlay = document.getElementById("drawer-overlay");
    const closeDrawerBtn = document.getElementById("close-drawer-btn");
    const inspectSource = document.getElementById("inspect-source");
    const inspectPage = document.getElementById("inspect-page");
    const inspectScore = document.getElementById("inspect-score");
    const inspectText = document.getElementById("inspect-text");

    // ── State variables ──
    let isUploading = false;
    let isChatting = false;

    // Initialize application state
    loadIndexedDocuments();

    // ── Document Operations ──

    // Load and list documents currently indexed in vector DB
    async function loadIndexedDocuments() {
        try {
            const resp = await fetch("/api/documents");
            if (!resp.ok) throw new Error("Could not retrieve documents list.");
            const docs = await resp.json();
            
            // Update badge count
            docCountBadge.innerText = docs.length;
            
            // Render list
            if (docs.length === 0) {
                guidelinesList.innerHTML = `<li class="empty-state">No files uploaded yet.</li>`;
                clearDbBtn.disabled = true;
            } else {
                clearDbBtn.disabled = false;
                guidelinesList.innerHTML = docs.map(doc => `
                    <li><i class="fa-solid fa-file-pdf"></i> ${escapeHTML(doc)}</li>
                `).join("");
            }
        } catch (err) {
            console.error("Error loading guidelines list:", err);
        }
    }

    // Handle single PDF file upload to API
    async function uploadFile(file) {
        if (isUploading) return;
        isUploading = true;
        
        showUploadStatus("Uploading & indexing in FAISS vector store...", "loading");

        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData
            });

            const data = await resp.json();

            if (!resp.ok) {
                const errMsg = data.detail || "Upload failed.";
                throw new Error(errMsg);
            }

            showUploadStatus(`Indexed successfully! ${data.chunks_created} chunks added.`, "success");
            loadIndexedDocuments();
        } catch (err) {
            console.error("Upload error:", err);
            showUploadStatus(`Error: ${err.message}`, "error");
        } finally {
            isUploading = false;
            fileInput.value = ""; // Reset
        }
    }

    // Clear whole database
    clearDbBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to delete all indexed PDFs and reset the vector store? This cannot be undone.")) return;
        
        try {
            const resp = await fetch("/api/documents", {
                method: "DELETE"
            });
            const data = await resp.json();
            
            if (!resp.ok) throw new Error(data.detail || "Failed to reset.");
            
            showUploadStatus("Database cleared successfully.", "success");
            loadIndexedDocuments();
            
            // Clear chat log
            messagesList.innerHTML = "";
            messagesList.classList.add("hidden");
            welcomeContainer.classList.remove("hidden");
        } catch (err) {
            console.error("Clear DB error:", err);
            alert(`Error clearing database: ${err.message}`);
        }
    });

    // ── Drag & Drop Event Handlers ──
    dropZone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });

    function showUploadStatus(msg, type) {
        uploadStatus.innerText = msg;
        uploadStatus.className = `status-msg ${type}`;
        uploadStatus.classList.remove("hidden");
        if (type === "success" || type === "error") {
            setTimeout(() => {
                uploadStatus.classList.add("hidden");
            }, 6000);
        }
    }

    // ── Chat Queries ──

    // Enable/Disable buttons based on input
    chatInput.addEventListener("input", () => {
        const text = chatInput.value.trim();
        sendBtn.disabled = text.length === 0 || isChatting;
        
        // Auto-grow height of text box
        chatInput.style.height = "auto";
        chatInput.style.height = (chatInput.scrollHeight - 4) + "px";
    });

    // Handle send click
    sendBtn.addEventListener("click", () => submitQuery());
    
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });

    // Handle suggested clicks
    suggestBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            chatInput.value = query;
            chatInput.dispatchEvent(new Event("input"));
            submitQuery();
        });
    });

    async function submitQuery() {
        const queryText = chatInput.value.trim();
        if (!queryText || isChatting) return;

        isChatting = true;
        sendBtn.disabled = true;
        chatInput.disabled = true;

        // Hide welcome cards and show messages frame
        welcomeContainer.classList.add("hidden");
        messagesList.classList.remove("hidden");

        // 1. Add User bubble
        appendMessage(queryText, "user");
        chatInput.value = "";
        chatInput.style.height = "auto";

        // Scroll to bottom
        scrollToBottom();

        // 2. Add Bot loader bubble
        const botBubble = appendMessage(`<i class="fa-solid fa-spinner fa-spin"></i> Generating clinical response...`, "bot", true);
        scrollToBottom();

        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: queryText })
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Server error during chat query.");
            }

            // 3. Render Answer with citations
            renderBotAnswer(botBubble, data);

        } catch (err) {
            console.error("Chat error:", err);
            botBubble.querySelector(".msg-bubble").innerHTML = `
                <span style="color: var(--danger-color);"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${escapeHTML(err.message)}</span>
            `;
        } finally {
            isChatting = false;
            chatInput.disabled = false;
            chatInput.focus();
            sendBtn.disabled = chatInput.value.trim().length === 0;
            scrollToBottom();
        }
    }

    function appendMessage(content, sender, isHTML = false) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `msg ${sender}`;
        
        const header = document.createElement("div");
        header.className = "msg-header";
        header.innerText = sender === "user" ? "Clinician Query" : "Clinical Assistant";
        msgDiv.appendChild(header);

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        if (isHTML) {
            bubble.innerHTML = content;
        } else {
            bubble.innerText = content;
        }
        msgDiv.appendChild(bubble);

        messagesList.appendChild(msgDiv);
        return msgDiv;
    }

    function renderBotAnswer(bubbleElement, chatResponse) {
        const bubble = bubbleElement.querySelector(".msg-bubble");
        bubble.innerHTML = ""; // Clear loader text

        // 1. Confidence Badge
        const badge = document.createElement("div");
        const level = chatResponse.confidence_level || "NONE";
        let levelClass = "level-none";
        let icon = "fa-circle-question";
        if (level.includes("HIGH")) {
            levelClass = "level-high";
            icon = "fa-circle-check";
        } else if (level.includes("MEDIUM")) {
            levelClass = "level-medium";
            icon = "fa-circle-info";
        } else if (level.includes("LOW")) {
            levelClass = "level-low";
            icon = "fa-triangle-exclamation";
        } else if (level.includes("NONE")) {
            levelClass = "level-none";
            icon = "fa-shield-halved";
        }
        
        badge.className = `confidence-indicator ${levelClass}`;
        const scoreText = (chatResponse.confidence_score !== undefined && chatResponse.confidence_score > -9.9) ? ` (score: ${chatResponse.confidence_score.toFixed(2)})` : '';
        badge.innerHTML = `<i class="fa-solid ${icon}"></i> Confidence: ${level}${scoreText}`;
        bubble.appendChild(badge);

        // 2. Answer body text (rendered as markdown)
        const textElement = document.createElement("div");
        textElement.className = "answer-body";
        textElement.innerHTML = renderMarkdown(chatResponse.answer);
        bubble.appendChild(textElement);

        // 3. Citations chips at the bottom
        if (chatResponse.citations && chatResponse.citations.length > 0) {
            const citHeader = document.createElement("div");
            citHeader.style.cssText = "font-size: 11px; margin-top: 12px; color: var(--text-muted); font-weight: 500;";
            citHeader.innerText = "Sources Cited (Click page link to inspect guidelines context):";
            bubble.appendChild(citHeader);

            const chipsContainer = document.createElement("div");
            chipsContainer.style.cssText = "display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;";

            chatResponse.citations.forEach((cite, index) => {
                const chip = document.createElement("span");
                chip.className = "citation-tag";
                chip.innerHTML = `<i class="fa-solid fa-arrow-up-right-from-square"></i> ${escapeHTML(cite.source)} (Pg ${cite.page_number})`;
                
                // Clicking opens details in Context Inspector
                chip.addEventListener("click", () => {
                    openInspector(cite);
                });
                
                chipsContainer.appendChild(chip);
            });

            bubble.appendChild(chipsContainer);
        }

        // 4. Feedback rating footer
        const footer = document.createElement("div");
        footer.className = "msg-footer";

        const fContainer = document.createElement("div");
        fContainer.className = "feedback-container";
        fContainer.innerHTML = `
            <span class="feedback-label">Helpful?</span>
            <button class="feedback-btn upvote-btn" title="Accurate & Grounded"><i class="fa-regular fa-thumbs-up"></i></button>
            <button class="feedback-btn downvote-btn" title="Inaccurate or Vague"><i class="fa-regular fa-thumbs-down"></i></button>
        `;

        const upBtn = fContainer.querySelector(".upvote-btn");
        const downBtn = fContainer.querySelector(".downvote-btn");

        const sendFeedback = async (rating) => {
            try {
                upBtn.disabled = true;
                downBtn.disabled = true;
                
                const response = await fetch("/api/feedback", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        query: chatResponse.query,
                        answer: chatResponse.answer,
                        rating: rating,
                        reason: rating === 1 ? "Upvoted" : "Downvoted"
                    })
                });
                
                if (response.ok) {
                    if (rating === 1) {
                        upBtn.className = "feedback-btn active-up";
                        upBtn.disabled = false;
                        downBtn.style.display = "none";
                    } else {
                        downBtn.className = "feedback-btn active-down";
                        downBtn.disabled = false;
                        upBtn.style.display = "none";
                    }
                }
            } catch (err) {
                console.error("Failed to send feedback:", err);
                upBtn.disabled = false;
                downBtn.disabled = false;
            }
        };

        upBtn.addEventListener("click", () => sendFeedback(1));
        downBtn.addEventListener("click", () => sendFeedback(-1));

        footer.appendChild(fContainer);
        bubble.appendChild(footer);
    }

    function scrollToBottom() {
        chatFrame.scrollTop = chatFrame.scrollHeight;
    }

    // ── Context Inspector Slider ──

    function openInspector(citation) {
        inspectSource.innerText = citation.source;
        inspectPage.innerText = `Page ${citation.page_number}`;
        inspectScore.innerText = citation.score.toFixed(4);
        inspectText.innerText = citation.text;

        // Slide drawer open
        citationDrawer.classList.add("open");
        drawerOverlay.classList.add("active");
    }

    function closeInspector() {
        citationDrawer.classList.remove("open");
        drawerOverlay.classList.remove("active");
    }

    closeDrawerBtn.addEventListener("click", closeInspector);
    drawerOverlay.addEventListener("click", closeInspector);

    // Escape characters safely to avoid HTML Injection
    function escapeHTML(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Lightweight markdown → HTML renderer (bold, italic, links, line-breaks)
    function renderMarkdown(text) {
        if (!text) return "";
        // Escape HTML first to prevent injection
        let html = escapeHTML(text);
        // **bold**
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        // *italic*
        html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
        // [link text](url)
        html = html.replace(
            /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );
        // Newlines → <br>
        html = html.replace(/\n/g, "<br>");
        return html;
    }
});
