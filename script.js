document.addEventListener('DOMContentLoaded', () => {

    // ================= API CONFIG =================
    const API = window.location.hostname === "localhost"
        ? "http://localhost:5000"
        : "https://fairness-ai-project-1.onrender.com/"; // change later

    // ================= NAVBAR =================
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // ================= DEMO ELEMENTS =================
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('fileInput');

    const step1 = document.getElementById('demo-step-1');
    const step2 = document.getElementById('demo-step-2');
    const step3 = document.getElementById('demo-step-3');

    // ================= FILE UPLOAD =================
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', startDemo);

    // ================= CHART =================
    let myChart = null;

    function initChart() {
        if (myChart) return;

        const ctx = document.getElementById('approvalChart').getContext('2d');

        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";

        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Group A', 'Group B'],
                datasets: [
                    {
                        label: 'Approved',
                        data: [0, 0],
                        backgroundColor: 'rgba(59, 130, 246, 0.8)'
                    },
                    {
                        label: 'Rejected',
                        data: [0, 0],
                        backgroundColor: 'rgba(148, 163, 184, 0.3)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    // ================= MAIN LOGIC =================
    const startDemo = async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert("Upload a CSV file");
            return;
        }

        step1.classList.add('hidden');
        step2.classList.remove('hidden');

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${API}/analyze`, {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            step2.classList.add('hidden');
            step3.classList.remove('hidden');

            initChart();
            updateDashboard(data);

        } catch (err) {
            console.error(err);
            alert("Backend error. Check deployment.");
        }
    };

    // ================= DASHBOARD UPDATE =================
    const valDI = document.getElementById('val-di');
    const valEopp = document.getElementById('val-eopp');
    const valAcc = document.getElementById('val-acc');
    const overallStatus = document.getElementById('overall-status');
    const insightBox = document.getElementById('insight-box');

    function updateDashboard(data) {
        valAcc.textContent = data.accuracy + "%";
        valDI.textContent = data.disparate_impact;
        valEopp.textContent = data.equal_opportunity;

        if (data.disparate_impact < 0.8) {
            overallStatus.className = 'badge badge-error';
            overallStatus.innerHTML = 'High Bias Detected';
        } else {
            overallStatus.className = 'badge badge-success';
            overallStatus.innerHTML = 'Fair Model';
        }

        insightBox.innerHTML = `
            <h5>Analysis Result</h5>
            <p>Bias Score (DI): ${data.disparate_impact}</p>
            <p>Equal Opportunity: ${data.equal_opportunity}</p>
            <p>Accuracy: ${data.accuracy}%</p>
        `;

        // Update Chart
        myChart.data.datasets[0].data = data.approved;
        myChart.data.datasets[1].data = data.rejected;
        myChart.update();
    }

    // ================= MITIGATION TOGGLE =================
    const mitigationToggle = document.getElementById('mitigation-toggle');

    mitigationToggle.addEventListener('change', async (e) => {
        const enabled = e.target.checked;

        try {
            const res = await fetch(`${API}/mitigate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enable: enabled })
            });

            const data = await res.json();
            updateDashboard(data);

        } catch (err) {
            console.error(err);
        }
    });

});

<input type="file" id="fileInput" hidden />