const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

// ベースパスの設定
const BASE_PATH = '/tmu/research-2025/onomatopoeia';

// Middleware
app.use(express.json());

// ベースパス配下で静的ファイルを配信
app.use(BASE_PATH, express.static('.'));

// ルートパスでも配信（後方互換性のため）
app.use(express.static('.'));

// Ensure result directory exists
const resultDir = path.join(__dirname, 'result');
if (!fs.existsSync(resultDir)) {
    fs.mkdirSync(resultDir);
}

// Function to generate unique filename
function generateFilename(configName) {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    const milliseconds = String(now.getMilliseconds()).padStart(3, '0');
    
    // Generate random string (6 characters)
    const randomStr = Math.random().toString(36).substring(2, 8);
    
    const timestamp = `${year}-${month}-${day}T${hours}-${minutes}-${seconds}.${milliseconds}-${randomStr}`;
    return `${timestamp}_${configName}.json`;
}

// ベースパス配下のルート定義
function addRoutes(basePath = '') {
    // Route to serve survey configuration
    app.get(`${basePath}/survey_config`, (req, res) => {
        try {
            const configPath = path.join(__dirname, 'survey_context.json');
            
            if (!fs.existsSync(configPath)) {
                // Create default config if file doesn't exist
                const defaultConfig = {
                    "title": "Form Title",
                    "description": "Description",
                    "contact": "email@email.com",
                    "contents": [
                        {
                            "type": "section",
                            "title": "オノマトペについて",
                            "description": "description",
                            "contents": [
                                {
                                    "item": "項目1",
                                    "description": "",
                                    "type": "slider",
                                    "required": true,
                                    "content": {
                                        "max-title": "これ以上ないほど力をかける",
                                        "min-title": "全く力をかけない",
                                        "max-value": 100,
                                        "min-value": 0
                                    }
                                }
                            ]
                        }
                    ]
                };
                
                fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2));
            }
            
            const configData = fs.readFileSync(configPath, 'utf8');
            const config = JSON.parse(configData);
            res.json(config);
        } catch (error) {
            console.error('Error loading survey config:', error);
            res.status(500).json({ error: 'Failed to load survey configuration' });
        }
    });

    // Route to handle form submission
    app.post(`${basePath}/submit_survey`, (req, res) => {
        const { timestamp, responses, configName } = req.body;
        
        if (!timestamp || !responses) {
            return res.status(400).json({ error: 'Missing required data' });
        }
        
        try {
            const filename = generateFilename(configName || 'default');
            const filepath = path.join(resultDir, filename);
            
            const data = {
                timestamp: timestamp,
                configName: configName || 'default',
                responses: responses,
                submittedAt: new Date().toISOString(),
                filename: filename
            };
            
            fs.writeFileSync(filepath, JSON.stringify(data, null, 2), 'utf8');
            
            console.log(`Survey response saved to: ${filename}`);
            res.json({ 
                success: true, 
                filename: filename,
                message: 'Survey response saved successfully' 
            });
        } catch (error) {
            console.error('File save error:', error);
            res.status(500).json({ error: 'Failed to save response' });
        }
    });

    // Route to get latest response
    app.get(`${basePath}/latest_response`, (req, res) => {
        try {
            const files = fs.readdirSync(resultDir)
                .filter(file => file.endsWith('.json'))
                .map(file => ({
                    name: file,
                    time: fs.statSync(path.join(resultDir, file)).mtime
                }))
                .sort((a, b) => b.time - a.time);
            
            if (files.length === 0) {
                return res.json({ message: 'No responses found' });
            }
            
            const latestFile = files[0].name;
            const filepath = path.join(resultDir, latestFile);
            const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
            
            res.json(data);
        } catch (error) {
            console.error('Error reading latest response:', error);
            res.status(500).json({ error: 'Failed to retrieve response' });
        }
    });

    // Route to list all responses
    app.get(`${basePath}/list_responses`, (req, res) => {
        try {
            const files = fs.readdirSync(resultDir)
                .filter(file => file.endsWith('.json'))
                .map(file => {
                    const filepath = path.join(resultDir, file);
                    const stats = fs.statSync(filepath);
                    return {
                        filename: file,
                        created: stats.birthtime,
                        modified: stats.mtime,
                        size: stats.size
                    };
                })
                .sort((a, b) => b.created - a.created);
            
            res.json(files);
        } catch (error) {
            console.error('Error listing responses:', error);
            res.status(500).json({ error: 'Failed to list responses' });
        }
    });

    // Route to list available survey configs
    app.get(`${basePath}/list_configs`, (req, res) => {
        try {
            const publicDir = path.join(__dirname, 'public');
            
            if (!fs.existsSync(publicDir)) {
                fs.mkdirSync(publicDir);
            }
            
            const files = fs.readdirSync(publicDir)
                .filter(file => file.endsWith('.json'))
                .map(file => ({
                    filename: file,
                    name: file.replace('.json', ''),
                    url: `${basePath}/${file.replace('.json', '')}`
                }));
            
            res.json(files);
        } catch (error) {
            console.error('Error listing configs:', error);
            res.status(500).json({ error: 'Failed to list configurations' });
        }
    });

    // API route to get specific survey config
    app.get(`${basePath}/api/config/:configName`, (req, res) => {
        const configName = req.params.configName;
        
        try {
            const publicDir = path.join(__dirname, 'public');
            const configPath = path.join(publicDir, `${configName}.json`);
            
            if (!fs.existsSync(configPath)) {
                return res.status(404).json({ error: `Configuration "${configName}" not found` });
            }
            
            const configData = fs.readFileSync(configPath, 'utf8');
            const config = JSON.parse(configData);
            res.json(config);
        } catch (error) {
            console.error('Error loading specific config:', error);
            res.status(500).json({ error: 'Failed to load survey configuration' });
        }
    });

    // Dynamic route to serve survey with specific JSON config
    app.get(`${basePath}/:configName`, (req, res) => {
        const configName = req.params.configName;
        
        // Skip static files and API routes
        if (configName.includes('.') || 
            configName === 'survey_config' || 
            configName === 'submit_survey' || 
            configName === 'latest_response' ||
            configName === 'list_configs') {
            return res.status(404).send('Not found');
        }
        
        try {
            const publicDir = path.join(__dirname, 'public');
            const configPath = path.join(publicDir, `${configName}.json`);
            
            // Check if the JSON file exists
            if (!fs.existsSync(configPath)) {
                return res.status(404).send(`Survey configuration "${configName}" not found`);
            }
            
            // Serve the HTML file with a query parameter to indicate which config to load
            const htmlPath = path.join(__dirname, 'index.html');
            if (!fs.existsSync(htmlPath)) {
                return res.status(404).send('Survey form not found');
            }
            
            // Read HTML content and inject config name
            let htmlContent = fs.readFileSync(htmlPath, 'utf8');
            
            // Add a script tag to set the config name before the existing scripts
            const configScript = `
            <script>
                window.SURVEY_CONFIG_NAME = '${configName}';
                window.BASE_PATH = '${basePath}';
            </script>`;
            
            // Insert the script before the closing </head> tag
            htmlContent = htmlContent.replace('</head>', `${configScript}\n</head>`);
            
            res.send(htmlContent);
        } catch (error) {
            console.error('Error serving survey:', error);
            res.status(500).send('Internal server error');
        }
    });

    // Serve the main HTML file
    app.get(basePath || '/', (req, res) => {
        const htmlPath = path.join(__dirname, 'index.html');
        
        if (basePath) {
            // ベースパス配下の場合、BASE_PATHを注入
            let htmlContent = fs.readFileSync(htmlPath, 'utf8');
            const configScript = `
            <script>
                window.BASE_PATH = '${basePath}';
            </script>`;
            htmlContent = htmlContent.replace('</head>', `${configScript}\n</head>`);
            res.send(htmlContent);
        } else {
            res.sendFile(htmlPath);
        }
    });
}

// ベースパス配下とルートパス両方でルートを追加
addRoutes(BASE_PATH);
addRoutes('');

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Survey server running at http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log('  - http://localhost:3000/ (default survey)');
    console.log(`  - http://localhost:3000${BASE_PATH}/ (base path survey)`);
    console.log('  - http://localhost:3000/{config_name} (custom survey)');
    console.log(`  - http://localhost:3000${BASE_PATH}/{config_name} (base path custom survey)`);
    console.log('  - http://localhost:3000/list_configs (list available configs)');
    console.log('  - http://localhost:3000/list_responses (list saved responses)');
    console.log('  - http://localhost:3000/latest_response (get latest response)');
    console.log(`Results will be saved to: ${resultDir}`);
    console.log('Press Ctrl+C to stop the server');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down server...');
    process.exit(0);
});
