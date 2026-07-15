(function () {
  var link = document.getElementById("repoLink");
  if (!link) return;

  var parts = window.location.hostname.split(".");
  var owner = parts.length > 2 ? parts[0] : "";
  var repo = window.location.pathname.split("/").filter(Boolean)[0] || "same-girl-search";

  if (owner && window.location.hostname.endsWith("github.io")) {
    link.href = "https://github.com/" + owner + "/" + repo;
  }
})();

