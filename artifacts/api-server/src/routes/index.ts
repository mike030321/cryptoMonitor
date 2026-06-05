import { Router, type IRouter } from "express";
import healthRouter from "./health";
import cryptoRouter from "./crypto";

const router: IRouter = Router();

router.use(healthRouter);
router.use(cryptoRouter);

export default router;
