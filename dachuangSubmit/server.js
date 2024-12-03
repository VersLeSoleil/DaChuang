const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const cors = require('cors');  // 导入 CORS 中间件

const app = express();
const port = 3000;

// 使用 CORS 中间件，允许所有来源访问
app.use(cors());

// 设置 multer 文件存储
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/'); // 存储在本地 'uploads' 文件夹
    },
    filename: (req, file, cb) => {
        // 使用原始文件名保存
        cb(null, file.originalname);
    }
});

const upload = multer({ storage });

// 创建静态目录
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/results', express.static(path.join(__dirname, 'results')));  // 新增静态目录提供

// 接收图片上传并处理
app.post('/upload', upload.single('image'), (req, res) => {
    if (req.file) {
        const imageName = req.file.originalname.split('.')[0]; // 获取输入图片的名字（例如 '14_1'）

        const command = `E:/software/anaconda/envs/xai_thyroid/python main.py --config-path xAI_config.json --method GradCAM --image-path uploads/ --output-path results/ --output-numpy results/`;

        // 执行 Python 命令
        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error: ${error.message}`);
                return res.status(500).json({ error: 'Failed to process image', details: error.message });
            }
            if (stderr) {
                console.error(`Stderr: ${stderr}`);
                return res.status(500).json({ error: 'Error during processing', details: stderr });
            }

            console.log(`Stdout: ${stdout}`);
            res.json({
                message: 'File uploaded and processed successfully',
                fileName: req.file.originalname
            });
        });
    } else {
        res.status(400).json({ error: 'No file uploaded' });
    }
});

// 启动服务
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
