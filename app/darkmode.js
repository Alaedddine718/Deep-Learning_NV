(function(){
  function apply(mode){
    document.documentElement.dataset.theme = mode;
    localStorage.setItem('theme', mode);
  }
  document.addEventListener('toggle-dark', ()=>{
    const current = localStorage.getItem('theme') || 'light';
    apply(current === 'light' ? 'dark' : 'light');
  });
  document.getElementById('toggle-dark')?.addEventListener('click', ()=>{
    const current = localStorage.getItem('theme') || 'light';
    apply(current === 'light' ? 'dark' : 'light');
  });
  apply(localStorage.getItem('theme') || 'light');
})();
