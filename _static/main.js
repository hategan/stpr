window.addEventListener("scroll", function() {
    if (document.body.scrollTop > 10 || document.documentElement.scrollTop > 10) {
        document.getElementById("top_nav").classList.add("shadow");
    }
    else {
        document.getElementById("top_nav").classList.remove("shadow");
    }
});