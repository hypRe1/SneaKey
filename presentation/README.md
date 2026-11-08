# SneaKey pitch deck

A self-contained HTML slide deck (7 slides, 16:9). Everything — styles, the Agent Esc mascot and the animations — lives in `index.html`. Fonts load from Google Fonts, so present with an internet connection (it falls back to system fonts offline).

## Present

Open `index.html` in a browser (double-click it, or serve the repo with `python -m http.server 8000` and go to http://localhost:8000/presentation/).

| Key | Action |
|---|---|
| → / Space / PageDown / click right half | Next slide |
| ← / PageUp / click left half | Previous slide |
| Home / End | First / last slide |
| F | Fullscreen |
| N | Show or hide speaker notes |

The URL remembers the slide (`index.html#3` opens slide 3).

## Edit

- Each slide is one `<section class="slide">` in `index.html`, in order. Edit the text directly.
- Placeholders are in **[square brackets]** — search for `[` to find them all. Current values from the main README are in each slide's speaker notes (`<aside class="notes">`).
- Animations: any element with `class="anim"` animates in when its slide opens. `style="--d:.4s"` sets its delay; `--a:fade` changes rise to a fade.
- Demo video: on slide 5, replace the placeholder transcript with your video (there's a commented `<video>` example). Put the video file in this folder.
- Brand colours and fonts are CSS variables at the top of the `<style>` block.
