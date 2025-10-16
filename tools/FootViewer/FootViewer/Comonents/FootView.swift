//
//  FootView.swift
//  FootViewer
//
//  Created by Tomo Kikuchi on 2025/05/29.
//

import Foundation
import Cocoa

public enum FootType {
    case left
    case right
}

public class FootView: NSView {
    
    // MARK: - Properties
    private var pressureData: [Double] = Array(repeating: 0.0, count: 16)
    private let footType: FootType
    
    // 足の形状のパス
    private var footPath: NSBezierPath?
    
    // 16点の位置（足の形状内での相対位置）
    private var sensorPositions: [CGPoint] {
        let basePositions: [CGPoint] = [
            // つま先部分（上部）
            CGPoint(x: 0.3, y: 0.9),   // 1. 親指先端
            CGPoint(x: 0.5, y: 0.85),  // 2. 人差し指先端
            CGPoint(x: 0.65, y: 0.8),  // 3. 中指先端
            CGPoint(x: 0.75, y: 0.75), // 4. 小指側先端
            
            // 前足部
            CGPoint(x: 0.2, y: 0.7),   // 5. 親指付け根
            CGPoint(x: 0.4, y: 0.65),  // 6. 人差し指付け根
            CGPoint(x: 0.6, y: 0.6),   // 7. 中指付け根
            CGPoint(x: 0.8, y: 0.55),  // 8. 小指付け根
            
            // 中足部
            CGPoint(x: 0.25, y: 0.45), // 9. 内側中足部
            CGPoint(x: 0.5, y: 0.4),   // 10. 中央中足部
            CGPoint(x: 0.75, y: 0.35), // 11. 外側中足部
            
            // 後足部
            CGPoint(x: 0.3, y: 0.25),  // 12. 内側かかと前
            CGPoint(x: 0.5, y: 0.2),   // 13. 中央かかと前
            CGPoint(x: 0.7, y: 0.15),  // 14. 外側かかと前
            
            // かかと
            CGPoint(x: 0.4, y: 0.05),  // 15. 内側かかと
            CGPoint(x: 0.6, y: 0.05)   // 16. 外側かかと
        ]
        
        // 左足の場合はX座標を反転
        if footType == .left {
            return basePositions.map { CGPoint(x: 1.0 - $0.x, y: $0.y) }
        } else {
            return basePositions
        }
    }
    
    public init(frame frameRect: NSRect, footType: FootType = .right) {
        self.footType = footType
        super.init(frame: frameRect)
        setup()
    }
    
    public required init?(coder: NSCoder) {
        self.footType = .right  // デフォルトは右足
        super.init(coder: coder)
        setup()
    }
    
    private func setup() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.white.cgColor
    }
    
    // MARK: - Public Methods
    public func updatePressureData(_ data: [Double]) {
        guard data.count == 16 else {
            print("Warning: Expected 16 data points, got \(data.count)")
            return
        }
        
        // データを0.0-1.0の範囲にクランプ
        pressureData = data.map { max(0.0, min(1.0, $0)) }
        
        // ビューを再描画
        DispatchQueue.main.async {
            self.needsDisplay = true
        }
    }
    
    // MARK: - Drawing
    public override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        
        // 背景をクリア
        context.setFillColor(NSColor.white.cgColor)
        context.fill(bounds)
        
        // 足の輪郭を描画
        drawFootOutline()
        
        // 圧力データをヒートマップとして描画
        drawPressureHeatmap()
        
        // センサー位置を描画
        drawSensorPositions()
    }
    
    private func drawFootOutline() {
        let footPath = createFootPath()
        
        // 足の輪郭を描画
        NSColor.black.setStroke()
        footPath.lineWidth = 2.0
        footPath.stroke()
        
        // 足の内部を薄いグレーで塗りつぶし
        NSColor.lightGray.withAlphaComponent(0.1).setFill()
        footPath.fill()
    }
    
    private func createFootPath() -> NSBezierPath {
        let path = NSBezierPath()
        let rect = bounds.insetBy(dx: 20, dy: 20)
        
        // 足の形状を近似
        let width = rect.width
        let height = rect.height
        
        if footType == .right {
            // 右足の形状
            path.move(to: CGPoint(x: rect.minX + width * 0.3, y: rect.maxY))
            
            // つま先のカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.8, y: rect.maxY - height * 0.2),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.5, y: rect.maxY + height * 0.1),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.7, y: rect.maxY - height * 0.1))
            
            // 外側のカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.85, y: rect.maxY - height * 0.6),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.9, y: rect.maxY - height * 0.3),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.9, y: rect.maxY - height * 0.5))
            
            // かかとへのカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.6, y: rect.minY),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.8, y: rect.maxY - height * 0.8),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.7, y: rect.minY))
            
            // かかとの底
            path.line(to: CGPoint(x: rect.minX + width * 0.4, y: rect.minY))
            
            // 内側のカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.15, y: rect.maxY - height * 0.6),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.3, y: rect.minY),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.1, y: rect.maxY - height * 0.8))
            
            // つま先への内側カーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.3, y: rect.maxY),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.1, y: rect.maxY - height * 0.3),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.2, y: rect.maxY - height * 0.1))
        } else {
            // 左足の形状（右足を左右反転）
            path.move(to: CGPoint(x: rect.minX + width * 0.7, y: rect.maxY))
            
            // つま先のカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.2, y: rect.maxY - height * 0.2),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.5, y: rect.maxY + height * 0.1),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.3, y: rect.maxY - height * 0.1))
            
            // 外側のカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.15, y: rect.maxY - height * 0.6),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.1, y: rect.maxY - height * 0.3),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.1, y: rect.maxY - height * 0.5))
            
            // かかとへのカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.4, y: rect.minY),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.2, y: rect.maxY - height * 0.8),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.3, y: rect.minY))
            
            // かかとの底
            path.line(to: CGPoint(x: rect.minX + width * 0.6, y: rect.minY))
            
            // 内側のカーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.85, y: rect.maxY - height * 0.6),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.7, y: rect.minY),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.9, y: rect.maxY - height * 0.8))
            
            // つま先への内側カーブ
            path.curve(to: CGPoint(x: rect.minX + width * 0.7, y: rect.maxY),
                       controlPoint1: CGPoint(x: rect.minX + width * 0.9, y: rect.maxY - height * 0.3),
                       controlPoint2: CGPoint(x: rect.minX + width * 0.8, y: rect.maxY - height * 0.1))
        }
        
        path.close()
        return path
    }
    
    private func drawPressureHeatmap() {
        let rect = bounds.insetBy(dx: 20, dy: 20)
        
        for (index, pressure) in pressureData.enumerated() {
            let position = sensorPositions[index]
            let center = CGPoint(
                x: rect.minX + rect.width * position.x,
                y: rect.minY + rect.height * position.y
            )
            
            // 圧力に基づいて色を計算（青→緑→黄→赤）
            let color = colorForPressure(pressure)
            
            // 圧力に基づいて円のサイズを計算（サイズを大きく）
            let baseRadius: CGFloat = 25.0  // 15.0から25.0に増加
            let maxRadius: CGFloat = 45.0   // 30.0から45.0に増加
            let radius = baseRadius + (maxRadius - baseRadius) * CGFloat(pressure)
            
            // グラデーション効果のために複数の円を描画
            for i in stride(from: radius, to: 0, by: -2) {
                let alpha = 1.0 - (radius - i) / radius * 0.7
                color.withAlphaComponent(alpha).setFill()
                
                let circleRect = NSRect(
                    x: center.x - i/2,
                    y: center.y - i/2,
                    width: i,
                    height: i
                )
                
                let circlePath = NSBezierPath(ovalIn: circleRect)
                circlePath.fill()
            }
        }
    }
    
    private func drawSensorPositions() {
        let rect = bounds.insetBy(dx: 20, dy: 20)
        
        for (index, position) in sensorPositions.enumerated() {
            let center = CGPoint(
                x: rect.minX + rect.width * position.x,
                y: rect.minY + rect.height * position.y
            )
            
            // センサー番号を描画
            let numberString = "\(index + 1)"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: NSFont.boldSystemFont(ofSize: 10),
                .foregroundColor: NSColor.black
            ]
            
            let attributedString = NSAttributedString(string: numberString, attributes: attributes)
            let stringSize = attributedString.size()
            
            let drawPoint = CGPoint(
                x: center.x - stringSize.width / 2,
                y: center.y - stringSize.height / 2
            )
            
            attributedString.draw(at: drawPoint)
        }
    }
    
    private func colorForPressure(_ pressure: Double) -> NSColor {
        // HSBカラーモデルを使用してヒートマップカラーを生成
        // 青(240°) → 緑(120°) → 黄(60°) → 赤(0°)
        let hue = (1.0 - pressure) * 240.0 / 360.0  // 0-1の範囲を240-0度に変換
        let saturation: CGFloat = 0.8
        let brightness: CGFloat = 0.9
        
        return NSColor(hue: hue, saturation: saturation, brightness: brightness, alpha: 1.0)
    }
}

extension FootView {
    // MARK: - Convenience Methods
    
    /// テスト用のランダムデータを生成
    public func generateTestData() {
        let testData = (0..<16).map { _ in Double.random(in: 0.0...1.0) }
        updatePressureData(testData)
    }
    
    /// 特定のセンサーの圧力値を設定
    public func setPressure(_ pressure: Double, forSensor index: Int) {
        guard index >= 0 && index < 16 else { return }
        pressureData[index] = max(0.0, min(1.0, pressure))
        
        DispatchQueue.main.async {
            self.needsDisplay = true
        }
    }
    
    /// 現在の圧力データを取得
    public func getCurrentPressureData() -> [Double] {
        return pressureData
    }
    
    /// すべての圧力データをリセット
    public func resetPressureData() {
        pressureData = Array(repeating: 0.0, count: 16)
        
        DispatchQueue.main.async {
            self.needsDisplay = true
        }
    }
}
