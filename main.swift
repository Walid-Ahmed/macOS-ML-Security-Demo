// main.swift
import Foundation
import CoreML

class SecurityDaemon {
    private var model: MinimalMacOSAnomalyAE?
    private var isMonitoring = false
    
    func start() {
        print("🚀 Starting macOS Security Daemon...")
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MinimalMacOSAnomalyAE(configuration: config)
            print("✅ AI Model loaded successfully!")
            print("📊 Model Info:")
            print("   - Input: 'input' (1 × 10)")
            print("   - Output: 'linear_3' (1 × 10)")
            
        } catch {
            print("❌ Failed to load model: \(error)")
            return
        }
        
        // Start monitoring
        isMonitoring = true
        monitorLoop()
    }
    
    private func monitorLoop() {
        print("🔍 Starting security monitoring loop...")
        print("   Press Ctrl+C to stop monitoring")
        
        var cycle = 0
        while isMonitoring {
            cycle += 1
            print("\n--- Monitoring Cycle #\(cycle) ---")
            simulateSecurityCheck()
            
            // Wait 5 seconds between checks
            Thread.sleep(forTimeInterval: 5.0)
        }
    }
    
    private func simulateSecurityCheck() {
        let testProcesses = [
            ("Safari Browser", [1.0, 0.2, 0.1, 0.9, 0.1, 0.0, 0.0, 0.2, 0.0, 0.1]),
            ("Unknown App", [0.0, 0.7, 0.8, 0.4, 0.5, 0.3, 0.0, 0.4, 0.0, 0.4]),
            ("Suspicious Process", [0.0, 0.9, 0.9, 0.2, 0.8, 0.7, 1.0, 0.6, 1.0, 0.8])
        ]
        
        for (name, features) in testProcesses {
            if let analysis = analyzeProcess(features: features) {
                switch analysis.riskLevel {
                case .high:
                    print("🚨 BLOCK: \(name) - Score: \(String(format: "%.4f", analysis.confidence))")
                case .medium:
                    print("⚠️ REVIEW: \(name) - Score: \(String(format: "%.4f", analysis.confidence))")
                case .low:
                    print("✅ ALLOW: \(name) - Score: \(String(format: "%.4f", analysis.confidence))")
                }
            }
        }
    }
    
    private func analyzeProcess(features: [Double]) -> SecurityAnalysis? {
        guard let model = model else {
            print("❌ Model not loaded")
            return nil
        }
        
        do {
            // Create input array - using Float16 to match model
            let inputArray = try MLMultiArray(shape: [1, 10], dataType: .float16)
            for (index, feature) in features.enumerated() {
                inputArray[index] = NSNumber(value: Float(feature)) // Convert to Float for Float16
            }
            
            // Create input using the generated class
            let input = MinimalMacOSAnomalyAEInput(input: inputArray)
            
            // Make prediction - NOW USING linear_3 instead of output
            let output = try model.prediction(input: input)
            
            // ✅ CORRECT: Use linear_3 as the output name
            let reconstruction = output.linear_3
            
            // Calculate reconstruction error
            let error = calculateError(original: features, reconstructed: reconstruction)
            
            return determineSecurityAction(error: error)
            
        } catch {
            print("❌ Prediction failed: \(error)")
            return nil
        }
    }
    
    private func calculateError(original: [Double], reconstructed: MLMultiArray) -> Double {
        var totalError: Double = 0
        for i in 0..<10 {
            let originalVal = original[i]
            // Convert Float16 output back to Double for calculation
            let reconstructedVal = Double(reconstructed[i].floatValue)
            let error = (originalVal - reconstructedVal) * (originalVal - reconstructedVal)
            totalError += error
        }
        return totalError / 10.0
    }
    
    private func determineSecurityAction(error: Double) -> SecurityAnalysis {
        // Adjusted thresholds for your model's performance
        let riskLevel: RiskLevel
        let action: SecurityAction
        
        if error < 0.22 {
            riskLevel = .low
            action = .allow
        } else if error < 0.35 {
            riskLevel = .medium
            action = .review
        } else {
            riskLevel = .high
            action = .block
        }
        
        return SecurityAnalysis(
            riskLevel: riskLevel,
            action: action,
            confidence: error,
            reconstructionError: error
        )
    }
    
    func stop() {
        isMonitoring = false
        print("\n🛑 Security monitoring stopped")
    }
}

// Data structures
struct SecurityAnalysis {
    let riskLevel: RiskLevel
    let action: SecurityAction
    let confidence: Double
    let reconstructionError: Double
}

enum RiskLevel {
    case low, medium, high
}

enum SecurityAction {
    case allow, review, block
}

// MARK: - Application Entry Point
print("==========================================")
print("🔒 macOS AI Security Daemon")
print("==========================================")

let daemon = SecurityDaemon()

// Setup signal handlers for graceful shutdown
signal(SIGINT) { signal in
    print("\n🛑 Received interrupt signal (Ctrl+C)")
    daemon.stop()
    exit(0)
}

signal(SIGTERM) { signal in
    print("\n🛑 Received termination signal")
    daemon.stop()
    exit(0)
}

// Start the daemon
daemon.start()

// Keep the application running
dispatchMain()
