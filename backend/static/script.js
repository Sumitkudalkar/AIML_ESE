// Empty string = same origin. Works whether served via FastAPI (recommended)
// or if you still want to hit a separate server, change to "http://localhost:8000"
const API_URL = "";

const now = new Date();
const offset = now.getTimezoneOffset() * 60000;
const video = document.getElementById('webcam-video');
const statusText = document.getElementById('status-text');
let localDate = (new Date(now - offset)).toISOString().split("T")[0];

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("date-picker").value = localDate;
  loadData();
  updateClock();
  setInterval(loadData, 2000);
  setInterval(updateClock, 1000);
});

function updateClock() {
  document.getElementById("live-clock").textContent = new Date().toLocaleTimeString();
}

async function loadData() {
  const date = document.getElementById("date-picker").value;
  try {
    const [attRes, statsRes] = await Promise.all([
      fetch(`${API_URL}/attendance?date_filter=${date}`),
      fetch(`${API_URL}/stats?date_filter=${date}`)
    ]);

    const records = await attRes.json();
    const stats = await statsRes.json();

    document.getElementById("stat-present").textContent = stats.present || 0;
    document.getElementById("stat-absent").textContent = stats.absent || 0;
    document.getElementById("stat-total").textContent = stats.total || 0;

    const tbody = document.getElementById("table-body");
    tbody.innerHTML = "";
    records.forEach((r, i) => {
      const row = `<tr>
        <td>${i + 1}</td>
        <td>${r.name}</td>
        <td>${r.time}</td>
        <td>Present</td>
      </tr>`;
      tbody.insertAdjacentHTML('beforeend', row);
    });
  } catch (err) {
    console.error("Could not connect to server:", err);
    document.getElementById("stat-present").textContent = "—";
    document.getElementById("stat-absent").textContent = "—";
    document.getElementById("stat-total").textContent = "—";
    document.getElementById("table-body").innerHTML =
      `<tr><td colspan="4" style="color:salmon;text-align:center;padding:20px">
        ⚠️ Cannot reach server. Make sure FastAPI is running and open this page via http://localhost:8000
      </td></tr>`;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    document.getElementById('register-btn').disabled = false;
    document.getElementById('video-overlay').classList.add('hidden');
    statusText.innerText = "Camera ready. Please look straight at the camera.";
  } catch (err) {
    statusText.innerText = "Error accessing camera: " + err.message;
  }
}

async function captureAndRegister() {
  const username = document.getElementById('student-username').value.trim();
  if (!username) {
    alert("Please enter a username first.");
    return;
  }

  statusText.innerText = "Capturing... Please hold still.";
  document.getElementById('register-btn').disabled = true;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');

  let capturedImages = [];

  for (let i = 0; i < 30; i++) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    let base64Image = canvas.toDataURL('image/jpeg', 0.8);
    capturedImages.push(base64Image);
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  statusText.innerText = "Processing and registering on the server...";

  try {
    // FIX: was '/register_face' (relative), now correctly hits the FastAPI endpoint
    const response = await fetch(`${API_URL}/register_face`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, images: capturedImages })
    });

    const result = await response.json();
    if (result.status === "success") {
      statusText.innerText = `✓ Saved ${result.faces_saved} face images for "${result.username}". Re-train the model to activate.`;
      // Refresh stats so the new student shows up immediately
      loadData();
    } else {
      statusText.innerText = "Registration failed: " + (result.detail || "Unknown error");
      document.getElementById('register-btn').disabled = false;
    }
  } catch (err) {
    statusText.innerText = "Network error: " + err.message;
    document.getElementById('register-btn').disabled = false;
  }
}

function downloadCSV() {
  const date = document.getElementById("date-picker").value;
  window.location.href = `${API_URL}/export?date_filter=${date}`;
}
