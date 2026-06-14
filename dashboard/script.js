document.addEventListener('DOMContentLoaded', () => {
  const cards = document.querySelectorAll('.card-content, .hero-band, .card-soft-tinted');

  cards.forEach((card, index) => {
    card.animate(
      [
        { opacity: 0, transform: 'translateY(14px)' },
        { opacity: 1, transform: 'translateY(0)' },
      ],
      {
        duration: 460,
        delay: index * 45,
        easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)',
        fill: 'both',
      }
    );
  });
});