/** @type {import('next').NextConfig} */

const path = require('path');

const isGitHubPages = process.env.GITHUB_PAGES === 'true';
const repoName = process.env.REPO_NAME || 'sam-web-demo';

const nextConfig = {
    // Enable static export for GitHub Pages
    output: 'export',

    // Set basePath for GitHub Pages (https://<user>.github.io/<repo>/)
    basePath: isGitHubPages ? `/${repoName}` : '',

    // Asset prefix for GitHub Pages
    assetPrefix: isGitHubPages ? `/${repoName}/` : '',

    // Disable image optimization (requires server)
    images: {
        unoptimized: true,
    },

    webpack: (config) => {
        // See https://webpack.js.org/configuration/resolve/#resolvealias
        config.resolve.alias = {
            ...config.resolve.alias,
            "onnxruntime-web/all": path.join(__dirname, 'node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs'),
        }
        return config;
    },
}
module.exports = nextConfig
