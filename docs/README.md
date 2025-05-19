# Nüshu Character Retrieval-Augmented LLM Agent - GitHub Pages

This folder contains the GitHub Pages website for the Nüshu Character Retrieval-Augmented Large Language Model Agent project.

## Setting up GitHub Pages

To make this site available online:

1. Push this repository to GitHub
2. Go to your repository's settings on GitHub
3. Scroll down to the GitHub Pages section
4. Select the `main` or `master` branch and the `/docs` folder as the source
5. Click Save
6. Your site will be published at `https://[your-username].github.io/[repository-name]/`

## Local Development

To test this website locally:

```bash
# Install a simple HTTP server if you don't have one
python -m pip install http-server

# Navigate to the docs directory
cd /path/to/DLNLP_assignment_25/docs

# Start the server
python -m http.server 8000
```

Then open your browser and navigate to `http://localhost:8000`.

## Structure

- `index.html` - Home page
- `about.html` - Detailed information about the project
- `models.html` - Information about the models used
- `styles.css` - CSS styles for the site
- `scripts.js` - JavaScript for interactive elements

## Image Paths

Note that image paths in the HTML files are relative to the repository root, using paths like `../latex/images/...`. This is intentional for GitHub Pages deployment.

If you're moving this site elsewhere, you may need to adjust these paths.
