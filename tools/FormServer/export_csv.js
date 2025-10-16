const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

// Connect to database
const dbPath = path.join(__dirname, 'survey_responses.db');
const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY, (err) => {
    if (err) {
        console.error('❌ Error connecting to database:', err.message);
        process.exit(1);
    }
});

// Function to format and display the latest response
function displayLatestResponse() {
    db.get(
        'SELECT * FROM survey_responses ORDER BY created_at DESC LIMIT 1',
        (err, row) => {
            if (err) {
                console.error('❌ Database error:', err.message);
                db.close();
                process.exit(1);
            }
            
            if (!row) {
                console.log('📭 No survey responses found in database.');
                db.close();
                return;
            }
            
            console.log('📊 LATEST SURVEY RESPONSE');
            console.log('='.repeat(50));
            console.log(`🆔 Response ID: ${row.id}`);
            console.log(`📅 Submitted: ${row.timestamp}`);
            console.log(`💾 Saved to DB: ${row.created_at}`);
            console.log();
            console.log('📝 RESPONSES:');
            console.log('-'.repeat(30));
            
            try {
                const responses = JSON.parse(row.responses);
                
                Object.entries(responses).forEach(([question, answer]) => {
                    console.log(`🔹 ${question}:`);
                    
                    if (Array.isArray(answer)) {
                        // Handle checkbox responses (arrays)
                        if (answer.length === 0) {
                            console.log('   (未選択)');
                        } else {
                            answer.forEach(item => {
                                console.log(`   ✓ ${item}`);
                            });
                        }
                    } else if (answer === '') {
                        console.log('   (未入力)');
                    } else {
                        console.log(`   ${answer}`);
                    }
                    console.log();
                });
                
            } catch (parseError) {
                console.error('❌ Error parsing response data:', parseError.message);
                console.log('Raw response data:', row.responses);
            }
            
            console.log('='.repeat(50));
            
            db.close((closeErr) => {
                if (closeErr) {
                    console.error('❌ Error closing database:', closeErr.message);
                } else {
                    console.log('✅ Database connection closed successfully.');
                }
            });
        }
    );
}

// Function to display all responses count
function displayStats() {
    db.get(
        'SELECT COUNT(*) as total FROM survey_responses',
        (err, row) => {
            if (err) {
                console.error('❌ Error getting stats:', err.message);
                return;
            }
            
            console.log(`📈 Total responses in database: ${row.total}`);
            console.log();
            
            // Now display the latest response
            displayLatestResponse();
        }
    );
}

// --- CSV出力機能の追加 ---
function exportAllResponsesToCSV() {
    db.all('SELECT * FROM survey_responses ORDER BY id', (err, rows) => {
        if (err) {
            console.error('❌ Database error:', err.message);
            db.close();
            process.exit(1);
        }

        if (!rows || rows.length === 0) {
            console.log('📭 No survey responses found in database.');
            db.close();
            return;
        }

        // すべてのresponses(JSON)をパースして、ユニークな質問リストを作る
        let allQuestions = new Set();
        const parsedRows = rows.map(row => {
            let responses = {};
            try {
                responses = JSON.parse(row.responses);
                Object.keys(responses).forEach(q => allQuestions.add(q));
            } catch (e) {}
            return { ...row, responses };
        });
        allQuestions = Array.from(allQuestions);

        // CSVヘッダー
        const headers = ['id', 'timestamp', 'created_at', ...allQuestions];
        const csvLines = [headers.join(',')];

        // 各行をCSV形式に変換
        parsedRows.forEach(row => {
            const line = [
                row.id,
                `"${row.timestamp}"`,
                `"${row.created_at}"`,
                ...allQuestions.map(q => {
                    let ans = row.responses[q];
                    if (Array.isArray(ans)) return `"${ans.join(';')}"`;
                    if (ans === undefined || ans === null) return '';
                    return `"${String(ans).replace(/"/g, '""')}"`;
                })
            ];
            csvLines.push(line.join(','));
        });

        // ファイルに書き出し（BOM付きで保存）
        const csvPath = path.join(__dirname, 'responses_export.csv');
        const bom = '\uFEFF';
        fs.writeFileSync(csvPath, bom + csvLines.join('\n'), 'utf8');
        console.log(`✅ Exported all responses to ${csvPath} (UTF-8 with BOM)`);
        db.close();
    });
}

// --- コマンドライン引数で動作切り替え ---
if (!fs.existsSync(dbPath)) {
    console.log('❌ Database file not found. Make sure the server has been run at least once.');
    process.exit(1);
}

const arg = process.argv[2];
if (arg === 'export_csv') {
    exportAllResponsesToCSV();
} else {
    console.log('🔍 Checking latest survey response...');
    console.log();
    displayStats();
}
