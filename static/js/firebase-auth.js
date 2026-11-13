// Firebase Authentication for SignAI Vision Website

import("https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js")
    .then(async (appModule) => {

        const authModule = await import(
            "https://www.gstatic.com/firebasejs/11.6.0/firebase-auth.js"
        );

        const firebaseConfig = {
            apiKey: "AIzaSyBIaOnBF5Ak1uqjwh3rgKTkPOzvBccFqdA",
            authDomain: "signai-vision.firebaseapp.com",
            projectId: "signai-vision",
            storageBucket: "signai-vision.firebasestorage.app",
            messagingSenderId: "613563813031",
            appId: "1:613563813031:web:c7fd68fb801847c0fddf82"
        };

        const app = appModule.initializeApp(firebaseConfig);
        const auth = authModule.getAuth(app);

        // Make authentication available to other website scripts
        window.signAIAuth = auth;
        window.signAIGetToken = async function () {
            const user = auth.currentUser;

            if (!user) {
                throw new Error("Please log in first.");
            }

            return await authModule.getIdToken(user);
        };

        // Create login screen
        const overlay = document.createElement("div");

        overlay.id = "signAI-login-screen";

        overlay.innerHTML = `
            <div style="
                width: 340px;
                max-width: 90%;
                padding: 30px;
                background: white;
                border-radius: 14px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.2);
                text-align: center;
                color: #222;
            ">
                <h2>SignAI Vision</h2>
                <p>Sign in to continue</p>

                <input
                    id="signAI-email"
                    type="email"
                    placeholder="Email address"
                    style="width:100%;padding:12px;margin:10px 0;box-sizing:border-box;"
                >

                <input
                    id="signAI-password"
                    type="password"
                    placeholder="Password"
                    style="width:100%;padding:12px;margin:10px 0;box-sizing:border-box;"
                >

                <button
                    id="signAI-login-button"
                    style="width:100%;padding:12px;margin-top:12px;cursor:pointer;"
                >
                    Login
                </button>

                <p id="signAI-login-error" style="color:red;"></p>
            </div>
        `;

        Object.assign(overlay.style, {
            position: "fixed",
            inset: "0",
            background: "#101827",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: "999999"
        });

        document.body.appendChild(overlay);

        // Login button
        document.getElementById("signAI-login-button")
            .addEventListener("click", async () => {

                const email = document.getElementById("signAI-email").value.trim();
                const password = document.getElementById("signAI-password").value;

                const errorBox = document.getElementById("signAI-login-error");

                errorBox.textContent = "";

                if (!email || !password) {
                    errorBox.textContent = "Please enter your email and password.";
                    return;
                }

                try {
                    await authModule.signInWithEmailAndPassword(
                        auth,
                        email,
                        password
                    );
                } catch (error) {
                    errorBox.textContent = "Login failed. Check your email and password.";
                }
            });

        // Show website only after successful login
        authModule.onAuthStateChanged(auth, (user) => {

            if (user) {
                overlay.style.display = "none";
                console.log("Signed in:", user.email);
            } else {
                overlay.style.display = "flex";
            }

        });

    })
    .catch((error) => {
        console.error("Firebase initialization failed:", error);
        alert("Unable to connect to Firebase. Please refresh the page.");
    });
// Send recognized sentence to another registered user
document.addEventListener("DOMContentLoaded", () => {

    const sendButton = document.getElementById("sendMessageBtn");
    const recipientInput = document.getElementById("recipientEmail");
    const statusBox = document.getElementById("sendMessageStatus");
    const sentenceOutput = document.getElementById("sentenceOutput");

    if (!sendButton) return;

    sendButton.addEventListener("click", async () => {

        const recipientEmail = recipientInput.value.trim();
        const messageText = sentenceOutput.value.trim();

        if (!recipientEmail || !messageText) {
            statusBox.textContent =
                "Please enter a recipient email and recognize a sentence.";
            return;
        }

        try {
            sendButton.disabled = true;
            statusBox.textContent = "Sending message...";

            const token = await window.signAIGetToken();

            const response = await fetch("/send-message", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token
                },
                body: JSON.stringify({
                    recipientEmail: recipientEmail,
                    text: messageText
                })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Message could not be sent.");
            }

            statusBox.textContent = "Message sent successfully!";

        } catch (error) {
            console.error("Message sending error:", error);
            statusBox.textContent = error.message;
        } finally {
            sendButton.disabled = false;
        }

    });

});



