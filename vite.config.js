import { defineConfig } from 'vite'
import { resolve } from 'path'
import { copyFileSync, mkdirSync, existsSync } from 'fs'

// 手动复制静态 JS 文件的插件
function copyStaticFiles() {
  return {
    name: 'copy-static-files',
    writeBundle() {
      const distDir = resolve(__dirname, 'dist')
      const srcDir = resolve(__dirname, 'src')

      // 确保 dist 目录存在
      if (!existsSync(distDir)) {
        mkdirSync(distDir, { recursive: true })
      }

      // 复制 JS 文件
      const filesToCopy = ['app.js', 'voiceover-workflow.js']
      filesToCopy.forEach(file => {
        const src = resolve(srcDir, file)
        const dest = resolve(distDir, file)
        if (existsSync(src)) {
          copyFileSync(src, dest)
          console.log(`Copied ${file} to dist/`)
        }
      })
    }
  }
}

export default defineConfig({
  root: 'src',
  base: './',
  plugins: [copyStaticFiles()],
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
})
