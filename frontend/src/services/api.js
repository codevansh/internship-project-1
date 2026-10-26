const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function analyzeContract(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
    });

    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error("The analysis service returned an unreadable response.");
    }
    if (!response.ok) {
        throw new Error(data.detail || "Contract analysis failed.");
    }
    return data;
}
