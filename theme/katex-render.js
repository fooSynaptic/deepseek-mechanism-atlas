// KaTeX auto-render: $...$ / $$...$$ (matches IDE Preview Markdown math).
document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== "function") {
    return;
  }
  var root = document.querySelector("#content main");
  if (!root) {
    return;
  }
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    ignoredTags: [
      "script", "noscript", "style", "textarea", "pre", "code",
    ],
    throwOnError: false,
  });
});
