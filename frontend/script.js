document.getElementById('webtoon-form').addEventListener('submit', getWebtoonInfo);

function getWebtoonInfo(e) {
    e.preventDefault();

    const title = document.getElementById('title').value;
    const description = document.getElementById('description').value;
    const genres = Array.from(document.querySelectorAll('input[name="genre"]:checked')).map(input => input.value);
    const allGenres = ['action', 'comic', 'daily', 'drama', 'fantasy', 'historical', 'pure', 'sensibility', 'sports', 'thrill', 'story', 'episode', 'omnibus'];
    const genreBinary = allGenres.map(genre => genres.includes(genre) ? 1 : 0);

    fetch('http://localhost:5000/get_webtoon_info', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ new_title: title, new_description: description, genres: genreBinary })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predicted-rating').innerText = data.other_info.predicted_rating;
        if (Array.isArray(data.other_info.recommended_webtoons)) {
            document.getElementById('similar-webtoons').innerHTML = data.other_info.recommended_webtoons.map(webtoon => `<li>${webtoon}</li>`).join('');
        } else {
            console.log("'recommended_webtoons' is not an array");
        }
        document.getElementById('generated-plot').innerText = data.plot;
    })
    .catch(err => console.error('Error:', err));
}

function scrollToBottom() {
    window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: 'smooth'
    });
}
