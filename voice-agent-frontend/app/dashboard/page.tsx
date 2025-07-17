import { PrismaClient } from "@prisma/client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Phone,
  CheckCircle,
  XCircle,
  RotateCcw,
  AlertTriangle,
  Lightbulb,
  Heart,
  Brain,
  MessageSquare,
  Clock,
  BarChart3,
} from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

const prisma = new PrismaClient()

async function getCallData() {
  try {
    const calls = await prisma.callAnalysis.findMany({
      orderBy: { created_at: "desc" },
    })
    const followUps = await prisma.followUp.findMany({
      orderBy: { created_at: "desc" },
    })
    return { calls, followUps }
  } catch (error) {
    console.error("Error fetching data:", error)
    return { calls: [], followUps: [] }
  } finally {
    await prisma.$disconnect()
  }
}

function getSentimentIcon(sentiment: string) {
  switch (sentiment?.toLowerCase()) {
    case "positive":
      return <Heart className="h-4 w-4 text-green-500" />
    case "negative":
      return <XCircle className="h-4 w-4 text-red-500" />
    default:
      return <MessageSquare className="h-4 w-4 text-yellow-500" />
  }
}

function getSentimentBadge(sentiment: string) {
  switch (sentiment?.toLowerCase()) {
    case "positive":
      return <Badge className="bg-green-100 text-green-800 border-green-200">सकारात्मक</Badge>
    case "negative":
      return <Badge variant="destructive">नकारात्मक</Badge>
    default:
      return <Badge variant="secondary">तटस्थ</Badge>
  }
}

function getInterestLevelText(interestLevel: boolean) {
  return interestLevel ? "जिज्ञासा दिखाई" : "उदासीनता/भ्रम"
}

function getInterestLevelBadge(interestLevel: boolean) {
  return interestLevel ? (
    <Badge className="bg-blue-100 text-blue-800 border-blue-200">रुचि दिखाई</Badge>
  ) : (
    <Badge variant="outline">रुचि नहीं दिखाई</Badge>
  )
}

function getActionRecommendations(call: any) {
  const recommendations = []

  // Confusion in intro
  if (!call.intro_clarity) {
    recommendations.push({
      trigger: "परिचय में भ्रम",
      action: "सरल शुरुआती स्क्रिप्ट का उपयोग करें",
      icon: <MessageSquare className="h-4 w-4 text-blue-500" />,
      color: "bg-blue-50 border-blue-200",
    })
  }

  // Negative sentiment
  if (call.sentiment?.toLowerCase() === "negative") {
    recommendations.push({
      trigger: "नकारात्मक भावना",
      action: "नरम स्वर और सहानुभूतिपूर्ण वाक्यों का उपयोग करें",
      icon: <Heart className="h-4 w-4 text-pink-500" />,
      color: "bg-pink-50 border-pink-200",
    })
  }

  // Low interest level
  if (!call.interest_level) {
    recommendations.push({
      trigger: "कम रुचि स्तर",
      action: "FAQ को सरल भाषा में दोबारा लिखें",
      icon: <Brain className="h-4 w-4 text-purple-500" />,
      color: "bg-purple-50 border-purple-200",
    })
  }

  // Follow-up needed
  if (call.outcome === "follow-up") {
    recommendations.push({
      trigger: "बाद में कॉल करने का अनुरोध",
      action: "फॉलो-अप मेमोरी लिस्ट में जोड़ें",
      icon: <Clock className="h-4 w-4 text-orange-500" />,
      color: "bg-orange-50 border-orange-200",
    })
  }

  return recommendations
}

function getOutcomeIcon(outcome: string) {
  switch (outcome) {
    case "success":
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case "follow-up":
      return <RotateCcw className="h-4 w-4 text-yellow-500" />
    default:
      return <XCircle className="h-4 w-4 text-red-500" />
  }
}

function getOutcomeText(outcome: string) {
  switch (outcome) {
    case "success":
      return "सफल"
    case "follow-up":
      return "फॉलो-अप आवश्यक"
    default:
      return "अस्वीकृत"
  }
}

function getOutcomeVariant(outcome: string): "default" | "secondary" | "destructive" | "outline" {
  switch (outcome) {
    case "success":
      return "default"
    case "follow-up":
      return "secondary"
    default:
      return "destructive"
  }
}

export default async function Dashboard() {
  const { calls, followUps } = await getCallData()
  const rejections = calls.filter((call: any) => call.outcome === "failure")
  const successful = calls.filter((call: any) => call.outcome === "success")
  const followUpCalls = calls.filter((call: any) => call.outcome === "follow-up")

  const stats = [
    {
      title: "कुल कॉल्स",
      value: calls.length,
      icon: Phone,
      description: "आज तक की गई कुल कॉल्स",
    },
    {
      title: "सफल कॉल्स",
      value: successful.length,
      icon: CheckCircle,
      description: "सफलतापूर्वक पूर्ण की गई कॉल्स",
    },
    {
      title: "फॉलो-अप",
      value: followUps.length,
      icon: RotateCcw,
      description: "फॉलो-अप की आवश्यकता वाली कॉल्स",
    },
    {
      title: "अस्वीकृत कॉल्स",
      value: rejections.length,
      icon: XCircle,
      description: "अस्वीकृत या असफल कॉल्स",
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">कॉल एनालिटिक्स डैशबोर्ड</h1>
        <p className="text-muted-foreground">विस्तृत कॉल विश्लेषण और कार्य सुझाव</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">{stat.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Detailed Call Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            विस्तृत कॉल विश्लेषण
          </CardTitle>
          <CardDescription>प्रत्येक कॉल का श्रेणीवार विश्लेषण और सुधार सुझाव</CardDescription>
        </CardHeader>
        <CardContent>
          {calls.length === 0 ? (
            <Alert>
              <AlertDescription>कोई कॉल डेटा उपलब्ध नहीं है। पहले कुछ कॉल्स करें।</AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-6">
              {calls.map((call: any) => {
                const recommendations = getActionRecommendations(call)
                let objections: string[] = []
                try {
                  objections = JSON.parse(call.objections || "[]")
                  if (!Array.isArray(objections)) objections = []
                } catch {
                  objections = []
                }

                return (
                  <Card key={call.call_id} className="relative">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg font-medium">{call.call_id}</CardTitle>
                        <Badge variant={getOutcomeVariant(call.outcome)} className="flex items-center gap-1">
                          {getOutcomeIcon(call.outcome)}
                          {getOutcomeText(call.outcome)}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(call.created_at).toLocaleString("hi-IN")}
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Analysis Categories */}
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Farmer Sentiment */}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            {getSentimentIcon(call.sentiment)}
                            <span className="text-sm font-medium">किसान की भावना</span>
                          </div>
                          {getSentimentBadge(call.sentiment)}
                          <p className="text-xs text-muted-foreground">स्वर में भावना का विश्लेषण</p>
                        </div>

                        {/* Interest Level */}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Brain className="h-4 w-4 text-blue-500" />
                            <span className="text-sm font-medium">रुचि स्तर</span>
                          </div>
                          {getInterestLevelBadge(call.interest_level)}
                          <p className="text-xs text-muted-foreground">{getInterestLevelText(call.interest_level)}</p>
                        </div>

                        {/* Intro Clarity */}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <MessageSquare className="h-4 w-4 text-green-500" />
                            <span className="text-sm font-medium">परिचय स्पष्टता</span>
                          </div>
                          <Badge variant={call.intro_clarity ? "default" : "destructive"}>
                            {call.intro_clarity ? "समझ गए" : "भ्रम में"}
                          </Badge>
                          <p className="text-xs text-muted-foreground">शुरुआती पिच की समझ</p>
                        </div>

                        {/* Objections */}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <AlertTriangle className="h-4 w-4 text-orange-500" />
                            <span className="text-sm font-medium">आपत्तियाँ</span>
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {objections.length > 0 ? (
                              objections.map((objection: string, index: number) => (
                                <Badge key={index} variant="outline" className="text-xs">
                                  {objection}
                                </Badge>
                              ))
                            ) : (
                              <Badge variant="secondary" className="text-xs">
                                कोई आपत्ति नहीं
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground">मूल्य, पात्रता, समय संबंधी चिंताएं</p>
                        </div>
                      </div>

                      <Separator />

                      {/* Action Recommendations */}
                      {recommendations.length > 0 && (
                        <div className="space-y-3">
                          <div className="flex items-center gap-2">
                            <Lightbulb className="h-4 w-4 text-yellow-500" />
                            <span className="text-sm font-medium">सुधार सुझाव</span>
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {recommendations.map((rec, index) => (
                              <div key={index} className={`p-3 rounded-lg border ${rec.color}`}>
                                <div className="flex items-start gap-2">
                                  {rec.icon}
                                  <div className="space-y-1">
                                    <p className="text-xs font-medium text-gray-700">{rec.trigger}</p>
                                    <p className="text-xs text-gray-600">{rec.action}</p>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Follow-ups Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RotateCcw className="h-5 w-5" />
            फॉलो-अप मेमोरी लिस्ट
          </CardTitle>
          <CardDescription>बाद में कॉल करने के अनुरोध वाली कॉल्स</CardDescription>
        </CardHeader>
        <CardContent>
          {followUps.length === 0 ? (
            <Alert>
              <AlertDescription>कोई फॉलो-अप कॉल उपलब्ध नहीं है।</AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {followUps.map((followUp: any) => (
                <Card key={followUp.call_id} className="border-yellow-200 bg-yellow-50">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">{followUp.call_id}</CardTitle>
                      <Badge variant="secondary" className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {followUp.status === "pending" ? "लंबित" : followUp.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-xs text-muted-foreground">
                      {new Date(followUp.created_at).toLocaleString("hi-IN")}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Performance Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            प्रदर्शन अंतर्दृष्टि
          </CardTitle>
          <CardDescription>कॉल गुणवत्ता और सुधार के क्षेत्र</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-600">
                {calls.length > 0 ? Math.round((successful.length / calls.length) * 100) : 0}%
              </div>
              <p className="text-sm text-green-700">सफलता दर</p>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">
                {calls.length > 0 ? Math.round((calls.filter((c: any) => c.intro_clarity).length / calls.length) * 100) : 0}%
              </div>
              <p className="text-sm text-blue-700">परिचय स्पष्टता दर</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">
                {calls.length > 0 ? Math.round((calls.filter((c: any) => c.interest_level).length / calls.length) * 100) : 0}
                %
              </div>
              <p className="text-sm text-purple-700">रुचि दर</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
