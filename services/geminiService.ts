/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, Modality } from "@google/genai";
import type { GenerateContentResponse } from "@google/genai";

// Lazily initialize the AI client to avoid crashing on load if API key is not set.
function getAi() {
    const API_KEY = process.env.API_KEY;
    if (!API_KEY) {
      // This allows the app to load and will only throw an error when a generation is attempted.
      throw new Error("API_KEY environment variable is not set.");
    }
    return new GoogleGenAI({ apiKey: API_KEY });
}


// --- Helper Functions ---

/**
 * Creates a fallback prompt to use when the primary one is blocked.
 * @param decade The decade string (e.g., "1950s").
 * @returns The fallback prompt string.
 */
function getFallbackPrompt(decade: string): string {
    return `صاوب تصويرة فوتوغرافية للشخص اللي فالتصويرة بحالا كان عايش ف ${decade}. التصويرة خاصها تبين الموضة، تسريحات الشعر، والجو العام ديال ديك الفترة. تأكد أن التصويرة النهائية واضحة وكتبان حقيقية لديك الحقبة.`;
}

/**
 * Extracts the decade (e.g., "1950s") from a prompt string.
 * @param prompt The original prompt.
 * @returns The decade string or null if not found.
 */
function extractDecade(prompt: string): string | null {
    const decadeMap: { [key: string]: string } = {
        'الخمسينات': '1950s',
        'الستينات': '1960s',
        'السبعينات': '1970s',
        'الثمانينات': '1980s',
        'التسعينات': '1990s',
        'الألفينات': '2000s',
    };
    for (const key in decadeMap) {
        if (prompt.includes(key)) {
            return key;
        }
    }
    return null;
}


/**
 * Processes the Gemini API response, extracting the image or throwing an error if none is found.
 * @param response The response from the generateContent call.
 * @returns A data URL string for the generated image.
 */
function processGeminiResponse(response: GenerateContentResponse): string {
    const imagePartFromResponse = response.candidates?.[0]?.content?.parts?.find(part => part.inlineData);

    if (imagePartFromResponse?.inlineData) {
        const { mimeType, data } = imagePartFromResponse.inlineData;
        return `data:${mimeType};base64,${data}`;
    }

    const textResponse = response.text;
    console.error("API did not return an image. Response:", textResponse);
    throw new Error(`الذكاء الاصطناعي جاوب بنص ماشي بتصويرة: "${textResponse || 'ما توصلنا بحتى شي نص.'}"`);
}

/**
 * A wrapper for the Gemini API call that includes a retry mechanism for internal server errors.
 * @param imagePart The image part of the request payload.
 * @param textPart The text part of the request payload.
 * @returns The GenerateContentResponse from the API.
 */
async function callGeminiWithRetry(imagePart: object, textPart: object): Promise<GenerateContentResponse> {
    const ai = getAi(); // Lazily get the AI instance.
    const maxRetries = 3;
    const initialDelay = 1000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: { parts: [imagePart, textPart] },
                config: {
                    responseModalities: [Modality.IMAGE],
                },
            });
        } catch (error) {
            console.error(`خطأ فاش عيطنا ل Gemini API (المحاولة ${attempt}/${maxRetries}):`, error);
            const errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
            const isInternalError = errorMessage.includes('"code":500') || errorMessage.includes('INTERNAL');

            if (isInternalError && attempt < maxRetries) {
                const delay = initialDelay * Math.pow(2, attempt - 1);
                console.log(`لقينا خطأ داخلي. غنعاودو المحاولة من بعد ${delay} ميلي ثانية...`);
                await new Promise(resolve => setTimeout(resolve, delay));
                continue;
            }
            throw error; // Re-throw if not a retriable error or if max retries are reached.
        }
    }
    // This should be unreachable due to the loop and throw logic above.
    throw new Error("Gemini API call failed after all retries.");
}


/**
 * Generates a decade-styled image from a source image and a prompt.
 * It includes a fallback mechanism for prompts that might be blocked in certain regions.
 * @param imageDataUrl A data URL string of the source image (e.g., 'data:image/png;base64,...').
 * @param prompt The prompt to guide the image generation.
 * @returns A promise that resolves to a base64-encoded image data URL of the generated image.
 */
export async function generateDecadeImage(imageDataUrl: string, prompt: string): Promise<string> {
  const match = imageDataUrl.match(/^data:(image\/\w+);base64,(.*)$/);
  if (!match) {
    throw new Error("صيغة الرابط ديال التصويرة غالطة. خاصها تكون 'data:image/...;base64,...'");
  }
  const [, mimeType, base64Data] = match;

    const imagePart = {
        inlineData: { mimeType, data: base64Data },
    };

    // --- First attempt with the original prompt ---
    try {
        console.log("كنحاولو نصاوبو التصويرة بالطلب الأصلي...");
        const textPart = { text: prompt };
        const response = await callGeminiWithRetry(imagePart, textPart);
        return processGeminiResponse(response);
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
        const isNoImageError = errorMessage.includes("الذكاء الاصطناعي جاوب بنص ماشي بتصويرة");

        if (isNoImageError) {
            console.warn("غالبا الطلب الأصلي تبلوكا. كنجربو دابا بطلب بديل.");
            const decade = extractDecade(prompt);
            if (!decade) {
                console.error("مقدرناش نجبدو الحقبة من الطلب، داكشي علاش منقدروش نخدمو بالبديل.");
                throw error; // Re-throw the original "no image" error.
            }

            // --- Second attempt with the fallback prompt ---
            try {
                const fallbackPrompt = getFallbackPrompt(decade);
                console.log(`كنحاولو نصاوبو التصويرة بالطلب البديل ديال ${decade}...`);
                const fallbackTextPart = { text: fallbackPrompt };
                const fallbackResponse = await callGeminiWithRetry(imagePart, fallbackTextPart);
                return processGeminiResponse(fallbackResponse);
            } catch (fallbackError) {
                console.error("حتى الطلب البديل منجحش.", fallbackError);
                const finalErrorMessage = fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
                throw new Error(`الذكاء الاصطناعي مقدرش يصاوب التصويرة لا بالطلب الأصلي لا بالبديل. آخر خطأ: ${finalErrorMessage}`);
            }
        } else {
            // This is for other errors, like a final internal server error after retries.
            console.error("وقع شي خطأ فادح فاش كنا كنصاوبو التصويرة.", error);
            throw new Error(`الذكاء الاصطناعي مقدرش يصاوب التصويرة. التفاصيل: ${errorMessage}`);
        }
    }
}