// KaTeX auto-render: $...$ / $$...$$ (matches IDE Preview Markdown math).
(function () {
  var opts = {
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
  };

  function renderAll() {
    if (typeof renderMathInElement !== "function") {
      return false;
    }
    var roots = [
      document.querySelector("#content main"),
      document.querySelector("nav.sidebar"),
      document.querySelector("#sidebar"),
    ].filter(Boolean);
    roots.forEach(function (root) {
      renderMathInElement(root, opts);
    });
    return true;
  }

  function waitAndRender(tries) {
    if (renderAll()) {
      return;
    }
    if (tries <= 0) {
      return;
    }
    setTimeout(function () {
      waitAndRender(tries - 1);
    }, 50);
  }

  document.addEventListener("DOMContentLoaded", function () {
    waitAndRender(40);
  });
})();
