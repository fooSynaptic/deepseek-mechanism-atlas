// Re-typeset after MathJax async load (mdBook pages with $...$ delimiters).
(function () {
  function typeset() {
    if (window.MathJax && window.MathJax.Hub) {
      MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    }
  }
  if (window.MathJax && window.MathJax.Hub && MathJax.Hub.Register) {
    MathJax.Hub.Register.StartupHook("End", typeset);
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      var n = 0;
      var t = setInterval(function () {
        if (window.MathJax && window.MathJax.Hub && MathJax.Hub.Register) {
          clearInterval(t);
          MathJax.Hub.Register.StartupHook("End", typeset);
        } else if (++n > 200) {
          clearInterval(t);
        }
      }, 50);
    });
  }
})();
