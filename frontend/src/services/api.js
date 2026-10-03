const API_BASE_URL = "http://127.0.0.1:8000";

export async function analyzeContract(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
    });
    
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail || "Contract analysis failed.");
    }
    return data;
}