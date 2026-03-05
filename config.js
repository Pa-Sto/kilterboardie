(() => {
  const PROD_API = "https://kilterboardie-api.fly.dev";

  const isLocal =
    window.location.protocol === "file:" ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1";

  window.KILTERBOARDIE_API = isLocal
    ? "http://127.0.0.1:8000"
    : PROD_API;

  window.KILTERBOARDIE_API_FALLBACK = isLocal
    ? ""
    : "https://kilterboardie-api.hybjf5nrbd.workers.dev";

  window.KILTERBOARDIE_SITE_BASE = isLocal
    ? ""
    : "https://raw.githubusercontent.com/Pa-Sto/kilterboardie/generated";
})();
