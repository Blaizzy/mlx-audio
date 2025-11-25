//
//  Kokoro-tts-lib
//
import Foundation

actor Timing {
  let id: String
  private let parent: Timing?

  private var start: UInt64
  private var finish: UInt64?
  private var childTasks: [Timing] = []
  private var delta: UInt64 = 0

  init(id: String, parent: Timing?) {
    self.id = id
    self.parent = parent
    self.start = DispatchTime.now().uptimeNanoseconds
  }

  func addChild(_ child: Timing) {
    childTasks.append(child)
  }

  func startTimer() {
    start = DispatchTime.now().uptimeNanoseconds
  }

  func stop() {
    let now = DispatchTime.now().uptimeNanoseconds
    finish = now
    delta += now - start
  }

  func log(spaces: Int = 0) async {
    guard finish != nil else { return }

    let spaceString = String(repeating: " ", count: spaces)
    Log.perf.debug("\(spaceString)\(self.id): \(self.deltaInSec) sec")
    for childTask in childTasks {
      await childTask.log(spaces: spaces + 2)
    }
  }

  var deltaTime: Double {
    Double(delta) / 1_000_000_000
  }

  var deltaInSec: String {
    String(format: "%.4f", Double(delta) / 1_000_000_000)
  }
}

actor BenchmarkTimer {
  static let shared = BenchmarkTimer()

  private var timers: [String: Timing] = [:]

  @discardableResult
  func create(id: String, parent parentId: String? = nil) async -> Timing? {
    guard timers[id] == nil else { return nil }

    var parentTiming: Timing?
    if let parentId {
      parentTiming = timers[parentId]
      guard parentTiming != nil else { return nil }
    }

    let timing = Timing(id: id, parent: parentTiming)
    if let parentTiming {
      await parentTiming.addChild(timing)
    }
    timers[id] = timing
    return timing
  }

  func stop(id: String) async {
    guard let timing = timers[id] else { return }
    await timing.stop()
  }

  func printLog(id: String) async {
    guard let timing = timers[id] else { return }
    await timing.log()
  }

  func reset() {
    timers = [:]
  }
}
