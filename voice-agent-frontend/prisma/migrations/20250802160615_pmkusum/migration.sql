-- CreateTable
CREATE TABLE "CallAnalysis" (
    "id" SERIAL NOT NULL,
    "call_id" TEXT NOT NULL,
    "sentiment" TEXT NOT NULL,
    "interest_level" BOOLEAN NOT NULL,
    "intro_clarity" BOOLEAN NOT NULL,
    "objections" JSONB NOT NULL,
    "outcome" TEXT NOT NULL,
    "language" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "CallAnalysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FollowUp" (
    "id" SERIAL NOT NULL,
    "call_id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "FollowUp_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "CallAnalysis_call_id_key" ON "CallAnalysis"("call_id");

-- CreateIndex
CREATE UNIQUE INDEX "FollowUp_call_id_key" ON "FollowUp"("call_id");
