const priceEl = document.getElementById('price');
const statusEl = document.getElementById('status');

async function fetchPrice() {
  try {
    const res = await fetch('/api/price');
    const data = await res.json();
    if (!data.ok || !data.amount) {
      throw new Error(data.error || 'No price');
    }
    priceEl.textContent = `$${Number(data.amount).toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
    const now = new Date();
    statusEl.textContent = `Updated: ${now.toLocaleTimeString()}`;
    statusEl.classList.remove('error');
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.classList.add('error');
  }
}

fetchPrice();
setInterval(fetchPrice, 500);
