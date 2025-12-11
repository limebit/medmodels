use tabled::{
    grid::{
        config::ColoredConfig,
        records::{ExactRecords, PeekableRecords, Records},
    },
    settings::TableOption,
};

// This code was mostly copied and adapted from https://docs.rs/tabled/0.20.0/tabled/settings/merge/struct.MergeDuplicatesVertical.html
// Feel free to use

#[derive(Debug)]
pub struct MergeDuplicatesVerticalByColumn {
    columns: Vec<usize>,
}

impl MergeDuplicatesVerticalByColumn {
    pub fn new(columns: Vec<usize>) -> Self {
        Self { columns }
    }
}

impl<R, D> TableOption<R, ColoredConfig, D> for MergeDuplicatesVerticalByColumn
where
    R: Records + PeekableRecords + ExactRecords,
{
    fn change(self, records: &mut R, cfg: &mut ColoredConfig, _: &mut D) {
        let count_rows = records.count_rows();
        let count_cols = records.count_columns();

        if count_rows == 0 || count_cols == 0 {
            return;
        }

        for column in self.columns {
            let mut repeat_length = 0;
            let mut repeat_value = String::new();
            let mut repeat_is_set = false;
            let mut last_is_row_span = false;
            for row in (0..count_rows).rev() {
                if last_is_row_span {
                    last_is_row_span = false;
                    continue;
                }

                // we need to mitigate messing existing spans
                let is_cell_visible = cfg.is_cell_visible((row, column).into());
                let is_row_span_cell = cfg.get_column_span((row, column).into()).is_some();

                if !repeat_is_set {
                    if !is_cell_visible {
                        continue;
                    }

                    if is_row_span_cell {
                        continue;
                    }

                    repeat_length = 1;
                    repeat_value = records.get_text((row, column).into()).to_owned();
                    repeat_is_set = true;
                    continue;
                }

                if is_row_span_cell {
                    repeat_is_set = false;
                    last_is_row_span = true;
                    continue;
                }

                if !is_cell_visible {
                    repeat_is_set = false;
                    continue;
                }

                let text = records.get_text((row, column).into());
                let is_duplicate = text == repeat_value;

                if is_duplicate {
                    repeat_length += 1;
                    continue;
                }

                if repeat_length > 1 {
                    cfg.set_row_span((row + 1, column).into(), repeat_length);
                }

                repeat_length = 1;
                repeat_value = records.get_text((row, column).into()).to_owned();
            }

            if repeat_length > 1 {
                cfg.set_row_span((0, column).into(), repeat_length);
            }
        }
    }
}
