async function pollJob(jobId, { onUpdate, onDone, intervalMs = 800 } = {}) {
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const data = await res.json();
      if (onUpdate) onUpdate(data);
      if (["done", "error", "cancelled"].includes(data.status)) {
        clearInterval(timer);
        if (onDone) onDone(data);
      }
    } catch (err) {
      console.error(err);
    }
  }, intervalMs);
  return timer;
}

async function cancelJob(jobId) {
  await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

function chartDefaults() {
  Chart.defaults.color = "#9bb0c5";
  Chart.defaults.borderColor = "#2a3441";
}
