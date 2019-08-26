window.addEventListener('load', function load() {
    window.removeEventListener('load', load);

    import('../wasm_build/index').then(({}) => {});
});
