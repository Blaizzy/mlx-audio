//
//  Kokoro-tts-lib
//
import Foundation

class BenchmarkTimer {
  class Timing {
    let id: String

    private var start: DispatchTime
    private var finish: DispatchTime?
    private var childTasks: [Timing] = []
    private let parent: Timing?
    private var delta: UInt64 = 0

    init(id: String, parent: Timing?) {
      start = DispatchTime.now()
      self.id = id
      self.parent = parent
      if let parent { parent.childTasks.append(self) }
    }

    func startTimer() {
      start = DispatchTime.now()
    }

    func stop() {
      finish = DispatchTime.now()
      delta += finish!.uptimeNanoseconds - start.uptimeNanoseconds
    }

    func log(spaces: Int = 0) {
      guard let _ = finish else { return }

      let spaceString = String(repeating: " ", count: spaces)
      print(spaceString + id + ": " + deltaInSec + " sec")
      for childTask in childTasks {
        childTask.log(spaces: spaces + 2)
      }
    }

    var deltaTime: Double { Double(delta) / 1_000_000_000 }
    var deltaInSec: String { String(format: "%.4f", Double(delta) / 1_000_000_000) }
  }

  static let shared = BenchmarkTimer()

  private init() {}

  private var timers: [String: Timing] = [:]

  @discardableResult
  func create(id: String, parent parentId: String? = nil) -> Timing? {
    guard timers[id] == nil else { return nil }

    var parentTiming: Timing?
    if let parentId {
      parentTiming = timers[parentId]
      guard parentTiming != nil else { return nil }
    }

    timers[id] = Timing(id: id, parent: parentTiming)
    return timers[id]
  }

  func stop(id: String) {
    guard let timing = timers[id] else { return }
    timing.stop()
  }

  func printLog(id: String) {
    guard let timing = timers[id] else { return }
    timing.log()
  }

  func reset() {
    timers = [:]
  }
}
