# Catalogue price audit

Date: 2026-11-16. Commit inspected: `585c301`. The target was not redefined and no model was refit because of this check.

## What the file supports

`usd_price_decimal_num` equals `sale_price_num` times `usd_currency_rate` on all 4,664 eligible rows, within 1%. The modeling target is that USD conversion of the recorded sale price.

`company_name` is the brand owner (Rolex SA, Compagnie Financière Richemont SA, Swatch Group AG). The auction house is the lot URL host: 2,115 Sotheby's lots, 2,548 Bonhams lots, and 1 `cars.bonhams.com` lot. Every row has a lot URL.

`hammer_price` is filled on 21 Sotheby's rows and on no Bonhams row. On those 21 rows the recorded sale is a constant multiple of the stored hammer:

| Year | Currency | Sale / stored hammer | Rows |
|---|---|---:|---:|
| 2024 | HKD | 1.20 | 1 |
| 2026 | HKD | 1.28 | 1 |
| 2026 | USD | 1.28 | 19 |

Of the 4,391 rows with a positive low and high estimate, 46.0% have a recorded USD price inside that range. The low is at or below the high on every such row. The file has no field for the date an estimate was written.

## Lots checked against the public page

| House | Lot | Recorded sale | Recorded estimates | What the public page showed | Status |
|---|---|---|---|---|---|
| Bonhams 26846, lot 66, 2021-05-05 | [page](https://www.bonhams.com/auction/26846/lot/66/cartier-18k-gold-tank-francaise-wristwatch/) | US$2,040 | US$1,000–2,000 | "Sold for US$2,040 inc. premium." | Verified. The recorded price includes premium. The extracted page did not restate the estimate. |
| Bonhams 26682, lot 49, 2021-02-23 | [page](https://www.bonhams.com/auction/26682/lot/49/) | £956 | £400–600 | "Sold for £956.25 inc. premium." | Verified as premium-inclusive. The file rounds to 956, a 0.25 difference. |
| Sotheby's, watches-online-7, lot 51, 2021-02-05 | [page](https://www.sothebys.com/en/buy/auction/2021/watches-online-7/a-platinum-and-diamond-set-wristwatch-circa-1920) | US$5,040 | US$4,000–8,000 | Estimate "4,000 - 8,000 USD." No sold price in the extracted text. | Estimate verified. Sale-price basis unverified. |
| Sotheby's, fine-watches-9, lot 66, 2026-03-31 | [page](https://www.sothebys.com/en/buy/auction/2026/fine-watches-9/reference-3730-calibre-de-cartier-diver-a-pink) | US$12,800, and a stored hammer of US$10,000 | US$7,000–11,000 | Estimate "7,000 - 11,000 USD." The result control says "Log in to view results." | Estimate verified. The public page does not show the hammer or the premium-inclusive total. The file's own two fields differ by the 1.28 factor. |

The 2026 Sotheby's lot is one of the 21 rows with both price fields. The first two Bonhams lots above are not in the 139-watch booster-only screen. The next section checks that screen.

## Booster-only flags

The frozen 2024 screen, unchanged at a 10% margin over the estimate midpoint, flags 139 validation watches that the linear model does not: 82 Sotheby's and 57 Bonhams. The four lots below are the first two such rows for each house in file order. They are a spot check, not a random sample. The same frozen fit that listed them also reprinted the 2024 scores: booster pinball $3,168.7295, linear pinball $3,975.7439, booster share 35.41%, linear share 30.42%. Training is the 2,490 sales through 2023-12-12. The 802 evaluation sales run from 2024-01-30 to 2024-12-19, and no record id is in both sets.

| House and lot | Sale date | Recorded estimates | Published estimates | Recorded target | Published price | Premium | Currency | Pre-auction estimate | Status |
|---|---|---|---|---|---|---|---|---|---|
| Bonhams 29150, lot 129. [URL](https://www.bonhams.com/auction/29150/lot/129/cartier-a-ladys-18k-gold-quartz-wristwatch-baignoire-ref-3208-circa-2010/) | 2024-02-21 | £4,000–£6,000, which the file converts to $5,050.95–$7,576.43 | Not in the extracted page | £5,376, which is $6,788.48 | "Sold for £5,376 inc. premium." No separate hammer | Verified. The recorded sale includes premium | GBP sale and GBP estimates use one file rate. $6,788.48 / £5,376 equals the rate that converts the estimate | Unverified. The closed page that was fetched does not show this lot's estimate | Sale verified. Estimate unverified |
| Bonhams 29150, lot 98. [URL](https://www.bonhams.com/auction/29150/lot/98/cartier-an-18k-gold-manual-wind-rectangular-form-wristwatch-cristallor-circa-1990/) | 2024-02-21 | Local estimate fields are blank. USD fields are $3,788.21–$5,050.95 | Not in the extracted page | £16,640, which is $21,011.96 | "Sold for £16,640 inc. premium." No separate hammer | Verified. The recorded sale includes premium | The sale conversion is consistent. The local estimate is missing while the USD estimate is filled | Unverified | Sale verified. Estimate fields are inconsistent inside the file, and the page did not show an estimate |
| Sotheby's fine-jewels-3, lot 239. [URL](https://www.sothebys.com/en/buy/auction/2024/fine-jewels-3/montre-bracelet-de-dame-or-baignoire-ref-3032) | 2024-03-27 | €5,000–€7,000, converted in the file to $5,416.18–$7,582.65 | "5,000 - 7,000 EUR" | €6,985, which is $7,566.40. Hammer field blank | Sold price not shown | Unverified | File converts sale and estimate with one rate. No outside FX check | The estimate is on the closed lot page. That is not a page saved before bidding | Estimate verified. Sale basis unverified |
| Sotheby's fine-watches, lot 450. [URL](https://www.sothebys.com/en/buy/auction/2024/fine-watches/calibre-de-cartier-reference-3300-montre-bracelet) | 2024-03-28 | €7,000–€9,000, converted in the file to $7,569.37–$9,732.04 | "7,000 - 9,000 EUR" | €6,350, which is $6,866.50. That sale is below the published low estimate. Hammer field blank | Sold price not shown | Unverified | Same file-rate treatment as lot 239 | The estimate is on the closed lot page. That is not a page saved before bidding | Estimate verified. Sale basis unverified. This lot is a booster-only flag even though the recorded sale is below the estimate, so the flag is not a statement that the watch sold above the catalogue range |

Both booster-only Bonhams pages use the same words as the earlier Bonhams checks: the recorded sale is the amount including premium. Neither page printed a separate hammer. A positive gap between that amount and the catalogue midpoint is not a profit. The file has no bid, buyer's invoice, resale, or cost of investigation. Lot 450 shows the distinction directly: the booster flagged it, and the recorded sale came in below the low estimate.

## What is inferred, and what is not

Inferred from the file, not from a live page: the same FX rate converts the sale price and, where checked, the estimate. The 2021 Sotheby's CHF lot converts an estimate of 6,000–10,000 at the same rate the file uses for the sale price (6,000 × 1.113053 = 6,678.32). FX rates were not checked against an outside source.

Not verified: that every Bonhams or Sotheby's row includes premium; that 1.20 and 1.28 are the houses' published premium schedules; that an estimate was on the page before bidding. Closed lot pages still display estimates. That is not a pre-sale snapshot. Presence of an estimate in the final file is not treated as proof it was available before bidding.

## What this changes

The screen compares a prediction of the recorded USD sale price with the catalogue estimate. On the lots that could be checked, those are not shown to be the same kind of amount: the recorded Bonhams price includes premium, and catalogue estimates are the published pre-sale ranges. A flag is a reason to investigate a lot. It is not a measured profit, and a 10% margin is not a premium adjustment.

This does not change the 2024 ranking on the recorded target. The booster's pinball remains $3,168.73 against the linear model's $3,975.74. It would change that ranking only if the team replaced the target with a hammer series and refit. That rescore is not recommended until both houses have a confirmed hammer, or a source note that states the basis. The 21 stored hammers are not that series.
