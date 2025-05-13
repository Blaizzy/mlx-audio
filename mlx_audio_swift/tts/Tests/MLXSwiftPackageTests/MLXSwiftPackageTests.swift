import XCTest
@testable import MySwiftPackage

final class MySwiftPackageTests: XCTestCase {
    func testGreet() {
        let myPackage = MySwiftPackage()
        let greeting = myPackage.greet()
        XCTAssertEqual(greeting, "Hello, World!")
    }
}