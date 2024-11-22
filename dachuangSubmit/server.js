const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const app = express();
const port = 3000;

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

// 获取上传文件夹中的图片列表
app.get('/uploads', (req, res) => {
    const uploadsDir = path.join(__dirname, 'uploads');

    fs.readdir(uploadsDir, (err, files) => {
        if (err) {
            return res.status(500).json({ error: '无法读取上传目录' });
        }
        // 过滤只包含图片文件的列表
        const images = files.filter(file => /\.(jpg|jpeg|png|gif|bmp)$/i.test(file));
        res.json(images);
    });
});

// 接收图片上传并处理
app.post('/upload', upload.single('image'), (req, res) => {
    if (req.file) {
        const imagePath = path.join(__dirname, 'uploads', req.file.originalname);
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
                fileName: req.file.originalname,
                results: stdout // 你可以根据需要调整返回内容
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
