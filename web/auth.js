// Shared by every page. One job: make sure a valid token is in
// localStorage before the page's own fetches run, and wrap fetch so a
// 401 anywhere sends you back to login instead of showing a broken page.
const API = "http://localhost:8000"; // change to the Render URL once deployed

function getToken() {
  return localStorage.getItem("trainò_token");
}

function requireAuthOrRedirect() {
  if (!getToken()) {
    window.location.href = "login.html";
    throw new Error("not logged in"); // stop the calling page's script here
  }
}

async function authedFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${getToken()}` },
  });
  if (res.status === 401) {
    localStorage.removeItem("trainò_token");
    window.location.href = "login.html";
    throw new Error("session expired");
  }
  return res;
}
