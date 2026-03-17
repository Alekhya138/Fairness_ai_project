const API = "https://fairness-ai-project-1.onrender.com";

function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    fetch(API + "/analyze", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        console.log(data);

        // update chart
        myChart.data.datasets[0].data = [
            data.male_rate * 100,
            data.female_rate * 100
        ];

        myChart.update();
    })
    .catch(err => console.error(err));
}