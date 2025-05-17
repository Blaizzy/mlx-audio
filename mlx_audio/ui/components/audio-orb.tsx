"use client"

import { useRef, useEffect } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import type * as THREE from "three"
import { Vector3, type MeshPhongMaterial, type MeshBasicMaterial, BackSide } from "three"
import { OrbitControls } from "@react-three/drei"

interface AudioOrbProps {
  isListening: boolean
  isProcessing: boolean
  audioUrl?: string
}

function OrbMesh({ isListening, isProcessing, audioUrl }: AudioOrbProps) {
  const sphereRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const pointLightRef = useRef<THREE.PointLight>(null)
  const originalVertices = useRef<Vector3[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const dataArrayRef = useRef<Uint8Array | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const rotationSpeedRef = useRef({ x: 0.001, y: 0.002 })

  // Set up the geometry and store original vertices
  useEffect(() => {
    if (sphereRef.current) {
      const geometry = sphereRef.current.geometry as THREE.BufferGeometry
      const positions = geometry.attributes.position

      // Store original vertices
      const vertices: Vector3[] = []
      for (let i = 0; i < positions.count; i++) {
        vertices.push(new Vector3(positions.getX(i), positions.getY(i), positions.getZ(i)))
      }
      originalVertices.current = vertices

      // Set up audio analyzer if we're in a browser environment
      if (typeof window !== "undefined") {
        try {
          const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
          analyserRef.current = audioContext.createAnalyser()
          analyserRef.current.fftSize = 256
          const bufferLength = analyserRef.current.frequencyBinCount
          dataArrayRef.current = new Uint8Array(bufferLength)

          // Initialize with some values for visualization
          if (dataArrayRef.current) {
            for (let i = 0; i < dataArrayRef.current.length; i++) {
              dataArrayRef.current[i] = 0
            }
          }
        } catch (error) {
          console.error("Audio API not supported:", error)
        }
      }
    }
  }, [])

  // Handle audio URL changes
  useEffect(() => {
    // We'll simulate audio data instead of trying to play actual audio
    // This avoids the "no supported source" error
  }, [audioUrl])

  // Update colors and animation parameters based on state
  useEffect(() => {
    if (sphereRef.current && glowRef.current) {
      const sphereMaterial = sphereRef.current.material as MeshPhongMaterial
      const glowMaterial = glowRef.current.material as MeshBasicMaterial

      if (isListening) {
        // Bright electric blue when listening
        sphereMaterial.color.set(0x00b3ff)
        sphereMaterial.emissive.set(0x0078cc)
        glowMaterial.color.set(0x00b3ff)
        rotationSpeedRef.current = { x: 0.002, y: 0.004 }

        // Start audio playback if we have an audio element
        if (audioElementRef.current) {
          audioElementRef.current.play().catch((e) => console.error("Audio playback failed:", e))
        }
      } else if (isProcessing) {
        // Vibrant cyan-blue when processing
        sphereMaterial.color.set(0x00d7ff)
        sphereMaterial.emissive.set(0x00a0cc)
        glowMaterial.color.set(0x00d7ff)
        rotationSpeedRef.current = { x: 0.003, y: 0.005 }
      } else {
        // Brighter default blue
        sphereMaterial.color.set(0x2196f3)
        sphereMaterial.emissive.set(0x1a6db8)
        glowMaterial.color.set(0x2196f3)
        rotationSpeedRef.current = { x: 0.001, y: 0.002 }

        // Pause audio playback if we have an audio element
        if (audioElementRef.current) {
          audioElementRef.current.pause()
        }
      }
    }
  }, [isListening, isProcessing])

  // Connect to microphone when listening
  useEffect(() => {
    let micStream: MediaStream | null = null
    let sourceNode: MediaStreamAudioSourceNode | null = null

    const connectMicrophone = async () => {
      if (!analyserRef.current || !isListening) return

      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const audioContext = analyserRef.current.context
        sourceNode = audioContext.createMediaStreamSource(micStream)
        sourceNode.connect(analyserRef.current)
      } catch (error) {
        console.error("Error accessing microphone:", error)
      }
    }

    if (isListening) {
      connectMicrophone()
    }

    return () => {
      if (micStream) {
        micStream.getTracks().forEach((track) => track.stop())
      }
      if (sourceNode) {
        sourceNode.disconnect()
      }
    }
  }, [isListening])

  // Helper function to get average frequency in a range
  const getAverageFrequency = (dataArray: Uint8Array, startIndex: number, endIndex: number) => {
    let sum = 0
    for (let i = startIndex; i <= endIndex; i++) {
      sum += dataArray[i]
    }
    return sum / (endIndex - startIndex + 1)
  }

  // Reset sphere to original state
  const resetSphere = () => {
    if (!sphereRef.current) return

    const geometry = sphereRef.current.geometry as THREE.BufferGeometry
    const positions = geometry.attributes.position

    for (let i = 0; i < positions.count; i++) {
      if (i >= originalVertices.current.length) continue

      const originalVertex = originalVertices.current[i]
      positions.setXYZ(i, originalVertex.x, originalVertex.y, originalVertex.z)
    }

    positions.needsUpdate = true
    geometry.computeVertexNormals()

    // Reset colors
    const sphereMaterial = sphereRef.current.material as MeshPhongMaterial
    sphereMaterial.color.set(0x0088ff)
    sphereMaterial.emissive.set(0x222222)

    if (glowRef.current) {
      const glowMaterial = glowRef.current.material as MeshBasicMaterial
      glowMaterial.color.set(0x0088ff)
    }
  }

  // Animation loop
  useFrame((state, delta) => {
    if (!sphereRef.current || !glowRef.current || originalVertices.current.length === 0) return

    try {
      // Get current time for pulsating effect
      const time = state.clock.elapsedTime

      // Rotate the orb
      sphereRef.current.rotation.y += rotationSpeedRef.current.y
      sphereRef.current.rotation.x += rotationSpeedRef.current.x
      glowRef.current.rotation.copy(sphereRef.current.rotation)

      // Simulate audio data when listening or processing
      if (isListening || isProcessing) {
        // Create simulated frequency data
        if (dataArrayRef.current) {
          for (let i = 0; i < dataArrayRef.current.length; i++) {
            // Create different patterns for different frequency ranges
            if (i < 10) {
              // Bass frequencies (0-9)
              dataArrayRef.current[i] = 128 + 127 * Math.sin(time * (1 + i * 0.1))
            } else if (i < 30) {
              // Mid frequencies (10-29)
              dataArrayRef.current[i] = 128 + 127 * Math.sin(time * 2 + i * 0.05)
            } else {
              // High frequencies (30+)
              dataArrayRef.current[i] = 128 + 127 * Math.sin(time * 3 + i * 0.02)
            }
          }
        }

        // Calculate average frequency values for different ranges
        const bassAvg = dataArrayRef.current ? getAverageFrequency(dataArrayRef.current, 0, 5) : 128
        const midAvg = dataArrayRef.current ? getAverageFrequency(dataArrayRef.current, 6, 20) : 128
        const trebleAvg = dataArrayRef.current ? getAverageFrequency(dataArrayRef.current, 21, 40) : 128

        // Calculate base pulsating factor
        const pulseFactor = Math.sin(time * 1.5) * 0.03 + 1 // Subtle pulsation (±3%)

        // Update sphere vertices based on frequency data
        const geometry = sphereRef.current.geometry as THREE.BufferGeometry
        const positions = geometry.attributes.position

        if (positions && positions.count > 0) {
          for (let i = 0; i < positions.count; i++) {
            if (i >= originalVertices.current.length) continue

            const originalVertex = originalVertices.current[i]
            if (!originalVertex) continue

            // Calculate normalized distance from center (0-1)
            const vertexLength = originalVertex.length()

            // Get frequency value based on vertex position
            let frequencyFactor = 0

            // Use different frequency ranges based on vertex position
            if (Math.abs(originalVertex.y) > vertexLength * 0.7) {
              // Top/bottom vertices - use treble
              frequencyFactor = trebleAvg / 255
            } else if (Math.abs(originalVertex.x) > vertexLength * 0.7) {
              // Left/right vertices - use mids
              frequencyFactor = midAvg / 255
            } else {
              // Other vertices - use bass
              frequencyFactor = bassAvg / 255
            }

            // Scale vertex based on both pulsation and frequency
            const scaleFactor = pulseFactor * (1 + frequencyFactor * 0.5)
            const distortedVertex = originalVertex.clone().multiplyScalar(scaleFactor)

            // Update position
            positions.setXYZ(i, distortedVertex.x, distortedVertex.y, distortedVertex.z)
          }

          // Mark positions for update
          positions.needsUpdate = true
          geometry.computeVertexNormals()
        }

        // Update colors based on frequency
        const hue = (bassAvg / 255) * 0.3
        const saturation = 0.8
        const lightness = 0.4 + (midAvg / 255) * 0.2

        const sphereMaterial = sphereRef.current.material as MeshPhongMaterial
        sphereMaterial.color.setHSL(hue, saturation, lightness)
        sphereMaterial.emissive.setHSL(hue, saturation, lightness * 0.5)

        // Update glow with both pulsation and audio reactivity
        const glowMaterial = glowRef.current.material as MeshBasicMaterial
        glowMaterial.color.setHSL(hue, saturation, lightness)

        const glowPulseFactor = 1 + Math.sin(time * 1.2) * 0.04
        glowRef.current.scale.set(
          glowPulseFactor * (1 + (bassAvg / 255) * 0.1),
          glowPulseFactor * (1 + (bassAvg / 255) * 0.1),
          glowPulseFactor * (1 + (bassAvg / 255) * 0.1),
        )

        // Update point light with both pulsation and audio reactivity
        if (pointLightRef.current) {
          const lightPulseFactor = 0.5 + Math.sin(time * 1.8) * 0.2
          pointLightRef.current.intensity = lightPulseFactor + (bassAvg / 255) * 1.5
          pointLightRef.current.color.setHSL(hue, saturation, lightness)
        }
      } else {
        // Apply subtle pulsating effect when no audio is playing
        const pulseFactor = Math.sin(time * 1.5) * 0.03 + 1 // Subtle pulsation (±3%)

        // Update sphere vertices for pulsating effect
        const geometry = sphereRef.current.geometry as THREE.BufferGeometry
        const positions = geometry.attributes.position

        if (positions && positions.count > 0) {
          for (let i = 0; i < positions.count; i++) {
            if (i >= originalVertices.current.length) continue

            const originalVertex = originalVertices.current[i]
            if (!originalVertex) continue

            positions.setXYZ(
              i,
              originalVertex.x * pulseFactor,
              originalVertex.y * pulseFactor,
              originalVertex.z * pulseFactor,
            )
          }

          // Mark positions for update
          positions.needsUpdate = true
          geometry.computeVertexNormals()
        }

        // Subtle color pulsation
        const hue = 0.6 // Blue hue
        const saturation = 0.8
        const lightness = 0.4 + Math.sin(time * 2) * 0.05 // Subtle brightness pulsation

        const sphereMaterial = sphereRef.current.material as MeshPhongMaterial
        sphereMaterial.color.setHSL(hue, saturation, lightness)
        sphereMaterial.emissive.setHSL(hue, saturation, lightness * 0.5)

        // Update glow with subtle pulsation
        const glowMaterial = glowRef.current.material as MeshBasicMaterial
        glowMaterial.color.setHSL(hue, saturation, lightness)

        glowRef.current.scale.set(
          1 + Math.sin(time * 1.2) * 0.04, // Slightly different frequency for interesting effect
          1 + Math.sin(time * 1.2) * 0.04,
          1 + Math.sin(time * 1.2) * 0.04,
        )

        // Subtle point light pulsation
        if (pointLightRef.current) {
          pointLightRef.current.intensity = 0.5 + Math.sin(time * 1.8) * 0.2
          pointLightRef.current.color.setHSL(hue, saturation, lightness)
        }
      }
    } catch (error) {
      console.error("Error in animation frame:", error)
    }
  })

  return (
    <>
      <mesh ref={glowRef} renderOrder={1}>
        <sphereGeometry args={[43, 32, 32]} />
        <meshBasicMaterial color={0x2196f3} transparent={true} opacity={0.2} side={BackSide} />
      </mesh>

      <mesh ref={sphereRef} renderOrder={2}>
        <icosahedronGeometry args={[40, 5]} />
        <meshPhongMaterial color={0x2196f3} emissive={0x1a6db8} shininess={50} flatShading={true} specular={0x777777} />
      </mesh>

      <pointLight ref={pointLightRef} position={[0, 0, 0]} intensity={1.7} distance={150} color={0x4db5ff} />
    </>
  )
}

function Scene({ isListening, isProcessing, audioUrl }: AudioOrbProps) {
  const { camera } = useThree()

  useEffect(() => {
    camera.position.set(0, 0, 120)
    camera.lookAt(0, 0, 0)
  }, [camera])

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[1, 1, 1]} intensity={0.9} />
      <OrbMesh isListening={isListening} isProcessing={isProcessing} audioUrl={audioUrl} />
      <OrbitControls enableZoom={false} enablePan={false} />
    </>
  )
}

export default function AudioOrb({ isListening, isProcessing, audioUrl }: AudioOrbProps) {
  return (
    <div className="w-full h-full">
      <Canvas>
        <Scene isListening={isListening} isProcessing={isProcessing} audioUrl={audioUrl} />
      </Canvas>
    </div>
  )
}
