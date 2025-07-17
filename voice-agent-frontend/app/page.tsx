"use client"

import type React from "react"

import { useState } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Phone, Loader2, CheckCircle, XCircle, Info } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export default function Home() {
  const [testNumber, setTestNumber] = useState("+1234567890")
  const [twilioNumber, setTwilioNumber] = useState("+0987654321")
  const [callId, setCallId] = useState("test_call_3")
  const [response, setResponse] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const apiUrl = "http://127.0.0.1:8001"
      const res = await axios.post(
        `${apiUrl}/api/start-call`,
        {
          call_id: callId,
          phone_number: testNumber,
        },
        {
          headers: { "Content-Type": "application/json" },
        },
      )

      setResponse(JSON.stringify(res.data, null, 2))
      toast({
        title: "कॉल सफलतापूर्वक शुरू की गई",
      })
      console.log("Call initiated:", res.data)
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message
      setError(errorMessage)
      toast({
        title: "कॉल शुरू करने में त्रुटि",
        description: errorMessage,
        variant: "destructive",
      })
      console.error("Error initiating call:", err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">PM-KUSUM Voice Agent</h1>
          <p className="text-muted-foreground">कॉल शुरू करें और कॉल एनालिटिक्स देखें</p>
        </div>
        <Badge variant="secondary" className="flex items-center gap-2">
          <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
          सिस्टम ऑनलाइन
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Phone className="h-5 w-5" />
              नई कॉल शुरू करें
            </CardTitle>
            <CardDescription>नीचे दिए गए फॉर्म को भरकर नई कॉल शुरू करें</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="testNumber">
                  टेस्ट नंबर <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="testNumber"
                  type="text"
                  value={testNumber}
                  onChange={(e) => setTestNumber(e.target.value)}
                  placeholder="+1234567890"
                  required
                />
                <p className="text-xs text-muted-foreground">E.164 प्रारूप में नंबर दर्ज करें (जैसे +1234567890)</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="twilioNumber">
                  ट्विलियो नंबर <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="twilioNumber"
                  type="text"
                  value={twilioNumber}
                  onChange={(e) => setTwilioNumber(e.target.value)}
                  placeholder="+0987654321"
                  required
                />
                <p className="text-xs text-muted-foreground">E.164 प्रारूप में ट्विलियो नंबर दर्ज करें</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="callId">
                  कॉल आईडी <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="callId"
                  type="text"
                  value={callId}
                  onChange={(e) => setCallId(e.target.value)}
                  placeholder="test_call_3"
                  required
                />
                <p className="text-xs text-muted-foreground">कॉल को ट्रैक करने के लिए यूनीक आईडी दर्ज करें</p>
              </div>

              <Button type="submit" disabled={loading} className="w-full" size="lg">
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    कॉल शुरू हो रही है...
                  </>
                ) : (
                  <>
                    <Phone className="mr-2 h-4 w-4" />
                    कॉल शुरू करें
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                टेस्ट निर्देश
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                  <span>टेस्ट नंबर पर कॉल का जवाब दें</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                  <span>सोलरबॉट का हिंदी प्रॉम्प्ट सुनें</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                  <span>मौखिक रूप से जवाब दें (जैसे, "हाँ" या "दोबारा कॉल")</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                  <span>FastAPI लॉग्स और डैशबोर्ड में परिणाम देखें</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {response && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                <div className="space-y-2">
                  <p className="font-medium">API Response:</p>
                  <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-32">{response}</pre>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertDescription>
                <div className="space-y-1">
                  <p className="font-medium">Error:</p>
                  <p className="text-sm">{error}</p>
                </div>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </div>
    </div>
  )
}
