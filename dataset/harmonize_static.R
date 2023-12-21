
harmonize_static_data <- function(static_fields_path, pus_field_path, out_path) {

    stat = readRDS(static_fields_path)
    pus = readRDS(pus_field_path)

    is_mach_id = "mach_ID" %in% names(stat)
    if (is_mach_id) {
        # Rename column
        names(stat)[names(stat) == "mach_ID"] <- "OBJECTID"
        names(pus)[names(pus) == "mach_ID"] <- "OBJECTID"
    }

    # All possible PUS classes.
    all_classes <- 1:29

    # Initialize a new data frame with the correct columns
    pus_new <- data.frame(matrix(NA, nrow = length(pus$mean), ncol = length(all_classes)))
    colnames(pus_new) <- all_classes

    # Fill the new data frame
    for (i in 1:length(pus$mean)) {
        # Extracting the names of the classes present in the current entry
        current_classes <- names(pus$mean[[i]])
        current_values <- pus$mean[[i]]

        # Assign values to the appropriate columns; missing classes will remain NA
        pus_new[i, current_classes] <- current_values
    }

    # Adding the OBJECTID column from the original data
    pus_new$OBJECTID <- pus$OBJECTID

    pus_new <- pus_new[,colSums(is.na(pus_new)) < nrow(pus_new)]
    pus_new[is.na(pus_new)] <- 0

    # Identify numeric columns
    numeric_cols <- sapply(pus_new, is.numeric)

    # Generate new column names for numeric columns
    new_col_names <- sprintf("pus%02d", as.numeric(names(pus_new)[numeric_cols]))

    # Replace the column names in the data frame
    names(pus_new)[numeric_cols] <- new_col_names

    # Drop unneeded fields and columns
    stat <- stat[stat$field.x == stat$field.y, ]
    stat <- stat[ , !(names(stat) %in% c("cov_static_prevah", "field.y"))]

    # Convert to wide format.
    stat_wide <- stats::reshape(stat,
                                idvar = "OBJECTID", 
                                timevar = "field.x", 
                                direction = "wide")

    # Rename the columns to make them more intuitive
    colnames(stat_wide) <- gsub("mean.", "", colnames(stat_wide))

    # Merge the wide version of stat with pus_new
    merged_df <- merge(pus_new, stat_wide, by = "OBJECTID", all.x = TRUE)

    if (is_mach_id) {
        # Rename column
        names(merged_df)[names(merged_df) == "OBJECTID"] <- "mach_ID"
    }

    # Save to file
    saveRDS(merged_df, file=out_path)
}

harmonize_static_data(
    static_fields_path="/data/william/data/RawFromMichael/obs/static/static_fields.rds",
    pus_field_path="/data/william/data/RawFromMichael/obs/static/pus_field.rds",
    out_path="/data/basil/static_harmonized/obs/harmonized.rds"
)

harmonize_static_data(
    static_fields_path="/data/william/data/RawFromMichael/prevah_307/static/static_fields.rds",
    pus_field_path="/data/william/data/RawFromMichael/prevah_307/static/pus_field.rds",
    out_path="/data/basil/static_harmonized/prevah_307/harmonized.rds"
)

