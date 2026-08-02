import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct ModelTypesTests {

    @Test func archConfigGemma4BaselineMatchesDocs() {
        let a = ArchConfig.gemma4_26B_A4B
        #expect(a.hiddenSize == 2816)
        #expect(a.intermediateSize == 2112)
        #expect(a.moeIntermediateSize == 704)
        #expect(a.numLayers == 30)
        #expect(a.numExperts == 128)
        #expect(a.topKExperts == 8)
        #expect(a.vocabSize == 262144)
        #expect(a.tieWordEmbeddings == true)
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.fullAttentionLayerMask.count == 30)
        let fullCount = a.fullAttentionLayerMask.reduce(0) { $0 + Int($1) }
        #expect(fullCount == 5, "Gemma 4 has 5 full-attention layers, got \(fullCount)")
        // Mask flags layers 5, 11, 17, 23, 29.
        for L in [5, 11, 17, 23, 29] {
            #expect(a.fullAttentionLayerMask[L] == 1, "layer \(L) should be full-attention")
        }
    }

    @Test func modelErrorDescriptionsContainKeyFacts() {
        let e1 = ModelError.archMismatch(field: "hiddenSize", expected: "2816", actual: "4096")
        #expect(e1.description.contains("2816") && e1.description.contains("4096"))
        let e2 = ModelError.unsupportedVersion(major: 2, minor: 0)
        #expect(e2.description.contains("2"))
        let e3 = ModelError.checksumMismatch(file: "model_weights.bin")
        #expect(e3.description.contains("model_weights.bin"))
    }

    @Test func qwen36_14B_BaselineUses90Experts() {
        let a = ArchConfig.qwen36_14B_A3B
        #expect(a.hiddenSize == 2048)
        #expect(a.numLayers == 40)
        #expect(a.numExperts == 90)
        #expect(a.topKExperts == 8)
        #expect(a.vocabSize == 248320)
        #expect(a.family == .qwen36)
        // Same linear/full attention pattern as the 35B.
        #expect(a.fullAttentionLayerMask == ArchConfig.qwen36_35B_A3B.fullAttentionLayerMask)
        #expect(a.linearAttention == ArchConfig.qwen36_35B_A3B.linearAttention)
    }

    @Test func baselineSelectionPicksVariantByNumExperts() {
        #expect(ArchConfig.baseline(for: .qwen36, numExperts: 90)?.numExperts == 90)
        #expect(ArchConfig.baseline(for: .qwen36, numExperts: 256)?.numExperts == 256)
        // Unknown expert count falls back to the first (default) baseline.
        #expect(ArchConfig.baseline(for: .qwen36, numExperts: 128)?.numExperts == 256)
        #expect(ArchConfig.baseline(for: .qwen36, numExperts: nil)?.numExperts == 256)
        // Gemma has one baseline; selection is stable.
        #expect(ArchConfig.baseline(for: .gemma4, numExperts: nil)?.numExperts == 128)
        #expect(ArchConfig.defaultBaseline(for: .qwen36)?.numExperts == 256)
    }
}
